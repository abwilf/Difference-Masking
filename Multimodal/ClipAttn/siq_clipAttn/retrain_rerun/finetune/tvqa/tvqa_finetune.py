"""
Finetunes TVQA.

I used a v3-32 for this (it's more expensive than VCR due to the use of video + sound data)
"""

import sys

sys.path.append('../../')
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder, MASK, encoder, AUDIOSPAN, MASKVIDEO
from finetune.common_dataloader import finetune_input_fn_builder, finetune_val_input_fn_builder
from mreserve.modeling import MerlotReserve, one_hot_pool, unit_normalize

from flax.training import train_state, early_stopping
from flax import jax_utils
import flax.linen as nn
from finetune.optimization import construct_finetuning_train_state, finetune_train_step
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32, f32_to_bf16, delete_prev_checkpoint
import argparse
import pandas as pd
import numpy as np
from flax.core.frozen_dict import freeze
from copy import deepcopy
import clu.parameter_overview
import functools
import time
import os
from dotenv import load_dotenv
import math
import logging
from jax.experimental.host_callback import call, id_print


load_dotenv('../../.env')

jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')

parser.add_argument(
    'pretrain_config_file',
    help='Where the config.yaml is located',
    type=str,
)
parser.add_argument(
    'ckpt',
    help='checkpoint to use',
    type=str,
)
parser.add_argument(
    '--lr',
    help='lr',
    type=float,
)
parser.add_argument(
    '--ne',
    help='ne',
    type=int,
    default=5,
)
parser.add_argument(
    '--output_grid_h',
    help='output_grid_h',
    type=int,
    default=12,
)
parser.add_argument(
    '--output_grid_w',
    help='output_grid_w',
    type=int,
    default=20,
)
parser.add_argument(
    '--output_name',
    help='output_name',
    type=str,
    default='',
)
parser.add_argument(
    '--wandb_name',
    help='wandb_name',
    type=str,
    default='merlotreserve-retrain-siq',
)
parser.add_argument(
    '--joint_proj',
    help='joint_projection',
    type=str, 
    choices=['joint_proj', 'no_proj'], 
    default='no_proj',
)
parser.add_argument(
    '--percent_data',
    help='percent_data',
    type=float,
    default=1.0,
)
parser.add_argument(
    '--alpha',
    help='alpha',
    type=float,
    default=0.0,
)
parser.add_argument(
    '--wandb_run_name',
    help='wandb_run_name',
    type=str,
    default='tvqa_1.0',
)
parser.add_argument(
    '--extn',
    help='extension',
    type=str,
    default='',
)
parser.add_argument(
    '--val_batch_size',
    help='val_batch_size -- defaults to 32',
    type=int,
    default=32,
)
parser.add_argument(
    '--output_ext',
    help='output_extension',
    action='store_true',
    default=False,
)
parser.add_argument(
    '--no_wandb',
    help='no_wandb',
    action='store_true',
    default=False,
)
parser.add_argument(
    '--mask_where',
    help='mask_where',
    type=str,
    default='random',
)
parser.add_argument(
    '-scan_minibatch',
    help='scan_minibatch -- basically, if this is true then batch size is 1 but we do gradient accumulation',
    action='store_true',
    default=False,
)
args = parser.parse_args()

if args.mask_where == "random":
    args.alpha = 0.0

# print(f"Loading from {args.config_file}", flush=True)
with open(args.pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

TOTAL_SIZE = 274 * 64
PERCENT_DATASET = args.percent_data
RECORD_PER_FILE = 274

seed_values = {'': 123456, '_run_1': 123456,'_run_2': 123457,'_run_3': 123458,'_run_4': 123459}

seed_value = seed_values[args.extn]

# config['data']['train_fns'] = os.path.join(os.environ["TFRECORDS_PATH"], "train{:03d}of256.tfrecord")
config['data']['train_fns'] = os.path.join(os.environ["TFRECORDS_PATH"], "train{:03d}of064.tfrecord")
config['data']['num_train_files'] = math.ceil((TOTAL_SIZE * PERCENT_DATASET) / RECORD_PER_FILE) # 256
config['data']['num_answers'] = 5
config['data']['random_scale_max'] = 1.1
config['data']['random_scale_min'] = 1.0
config['data']['num_segments'] = 7
config['data']['mask_where'] = args.mask_where
config['data']['alpha'] = args.alpha
config['data']['seed'] = seed_value

config['device']['batch_size'] = 8
config['device']['prefetch_size'] = 0
config['device']['n_fns_per_cycle'] = 64 #256

proj_stat = args.joint_proj #"joint_proj" if args.joint_proj else "no_proj"
    
ext = "tvqa_"+args.mask_where+"_"+ str(args.percent_data)+ "_"+str(args.output_grid_h)+'_'+str(args.output_grid_w)+'_'+str(args.alpha)+'_'+proj_stat+args.extn

NUM_EPOCH = args.ne
# TRAIN_SIZE = 122112



TRAIN_SIZE = config['data']['num_train_files'] * RECORD_PER_FILE if PERCENT_DATASET < 1.0 else TOTAL_SIZE
steps_per_epoch = TRAIN_SIZE // config['device']['batch_size']

config['optimizer'] = {
    'beta_2': 0.98,
    'eps': 1e-6,
    'learning_rate': args.lr,
    'num_train_steps': NUM_EPOCH * steps_per_epoch,
    'num_warmup_steps': int(0.5 * steps_per_epoch),
    'use_bfloat16_adam': True,
    'weight_decay_rate': 0.1,
    'do_bias_correction': True,
}

config['device']['iterations_per_loop'] = steps_per_epoch
config['data']['lang_seq_len'] = 256

cfg_name = args.pretrain_config_file.split('/')[-1]
seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

config['device']['output_dir'] = os.path.join(os.environ["OUTPUT_PATH"], cfg_name)
if args.output_name != '':
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], args.output_name)
if args.output_ext:
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], ext)
else:
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], seattle_time)

# import pdb; pdb.set_trace()
if os.path.exists(config['device']['output_dir']):
    print("It is already done!")
    sys.exit()

LOG = f"/home/sakter/results/retrain_siq/logs/{ext}.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.INFO, force=True)

np.random.seed(seed_value)
config['model']['output_grid'] = [args.output_grid_h, args.output_grid_w]
ds_train_iter = finetune_input_fn_builder(config, 'tvqa')
# _, dummy_batch = next(ds_train_iter)


config['_ckpt'] = args.ckpt
tags = [args.mask_where]
tags.append("TPU4") #### Need Editting
tags.append(str(args.percent_data))
tags.append("alpha="+str(args.alpha))
tags.append(proj_stat)
if args.output_name != '':
    tags.append(args.output_name)
if (jax.process_index() == 0) and not args.no_wandb:
    import wandb
    wandb_run_name = ext # args.wandb_run_name
    wandb.init(config=config, project=args.wandb_name, entity='snat', name=wandb_run_name, notes=f'Loaded from {cfg_name}', tags=tags)
else:
    wandb = None

class MerlotReserveTVQA(MerlotReserve):
    def setup(self):
        super().setup()
        self.proj = nn.Dense(features=1, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=0.02), name='proj',
                             use_bias=False)
        
        ########## EDITING/ ADDED W/O PERTURBATION
        # self.joint_proj = nn.Dense(features=self.config['hidden_size'], dtype=self.dtype,
        #                            kernel_init=kernel_init, name='head')
        ##########

    def __call__(self, batch):

        # Encode images (twice)
        batch_size, images_per_batch, seq_size, img_dim = batch['images'].shape # TensorShape([1, 7, 540, 768])
        mask_info = batch['masks_info'] #if (args.mask_where == "face" or args.mask_where == "body") else None
        vision_out = self.vision_encoder(batch['images'].reshape(batch_size * images_per_batch, seq_size, img_dim), mask = mask_info)
        imgs_enc = vision_out['seq_attnpool']
        imgs_enc = imgs_enc.reshape(batch_size, images_per_batch, seq_size // 4, self.hidden_size) # (1, 7, seq_size // 4, 768)
        
        num_ans_per = 1

        batch_size, num_ans_per, joint_seq_len, two_ = batch['textonly_seqs'].shape # (1, 1, 256, 2)
        imgs_enc = imgs_enc.reshape(batch_size, images_per_batch * seq_size // 4, self.hidden_size).repeat(num_ans_per, axis=0) # (1, 1080, 768)
       
        
        text_toks = batch['textonly_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len)
        vision_inputs_text = self.prepare_multimodal_inputs(
            tokens=text_toks,
            token_segment_idx=batch['textonly_seqs'][..., 1].reshape(batch_size * num_ans_per, joint_seq_len),
            vision_input=imgs_enc,
        )
        
        # Encode audio
        # Audio clips are provided as [batch_size, num_segments, num_audio_subsegments, audio_seq_len, num_mels]
        batch_size, num_segments, num_audio_subsegments, audio_seq_len, num_mels = batch['audio_clips'].shape # (1, 7, 3, 60, 65)
        audio_enc = self.audio_encoder(batch['audio_clips'].reshape(-1, audio_seq_len, num_mels))['seq_attnpool'] # (21, 6, 768)

        _, audio_token_len, hidden_size = audio_enc.shape # (21, 6, 768)
        num_audio_spans = num_segments * num_audio_subsegments # 21

        audio_enc = audio_enc.reshape(batch_size, num_audio_spans, audio_token_len, hidden_size) # (1, 21, 6, 768)
        audio_enc = audio_enc.repeat(num_ans_per, axis=0) # (1, 21, 6, 768)

        audio_toks = batch['audio_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len) # (1, 256) 
        audio_pointers = (jnp.cumsum((audio_toks == AUDIOSPAN).astype(jnp.int32), -1) - 1) // audio_token_len # (1, 256)
        audio_pointers = audio_pointers % num_audio_spans # (1, 256)
        
        vision_inputs_audio = self.prepare_multimodal_inputs(
            tokens=batch['audio_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len),
            token_segment_idx=batch['audio_seqs'][..., 1].reshape(batch_size * num_ans_per, joint_seq_len),
            vision_input=imgs_enc,
            audio_spans=audio_enc,
            audio_pointers=audio_pointers,
        )
        
        audio_toks = batch['vision_seqs_audio'][..., 0].reshape(batch_size * num_ans_per, -1) # (1,1201)
        text_toks = batch['vision_seqs_text'][..., 0].reshape(batch_size * num_ans_per, -1) # (1,1201)         
        
        real_bsizes = [vision_inputs_audio['x'].shape[0], vision_inputs_text['x'].shape[0]]
        x = jnp.concatenate([vision_inputs_audio['x'], vision_inputs_text['x']], 0) # (2,1201,768)
        coords = jnp.concatenate([vision_inputs_audio['rotary_coords'], vision_inputs_text['rotary_coords']], 0) # (2,1201,4)
        attnmask = jnp.concatenate([vision_inputs_audio['attention_mask'], vision_inputs_text['attention_mask']], 0) # (2,1201,1201)
        
        keys = ['audio2vision', 'text2vision']
        joint_enc = self.joint_transformer(x, rotary_coords=coords, attention_mask=attnmask)['seq'] # (2,1201,768)
        
        
        ########## EDITING/ ADDED W/O PERTURBATION
        if args.joint_proj == "joint_proj":
            joint_enc = self.joint_proj(joint_enc)
        ##########
        
        
        mm_outputs = {k: z for k, z in zip(keys, jnp.split(joint_enc, np.cumsum(real_bsizes), axis=0))}
        
        mm_outputs['audio2vision'] = mm_outputs['audio2vision'][:, joint_seq_len:] # (1, 945, 768)
        mm_outputs['text2vision'] = mm_outputs['text2vision'][:, joint_seq_len:] # (1, 945, 768)

        # Pool from the right tokens
        is_pool = (audio_toks == MASK)[:, joint_seq_len:] # (1,945)
        a2v_cumulative_idx = jnp.cumsum(is_pool.astype(jnp.int32), -1) - 1 # (1,945)
        
        a2v = one_hot_pool(is_pool,
               idx=a2v_cumulative_idx,
               v=mm_outputs['audio2vision'],
               num_segments=images_per_batch * seq_size // 4)['x'].reshape((batch_size * images_per_batch * seq_size // 4, self.hidden_size))
        # 'x': (1, 945, 768), 'idx_oh': (1,945,945), (945, 768)
        
        a2v_out = one_hot_pool(is_pool,
               idx=a2v_cumulative_idx,
               v=vision_out['target_seq_attnpool'].reshape(batch_size, images_per_batch * seq_size // 4, -1),
               num_segments=images_per_batch * seq_size // 4)['x'].reshape((batch_size * images_per_batch * seq_size // 4, self.hidden_size))
        
        is_pool = (text_toks == MASK)[:, joint_seq_len:] # (1, 945)
        t2v_cumulative_idx = jnp.cumsum(is_pool.astype(jnp.int32), -1) - 1 # (1, 945)
        
        t2v = one_hot_pool(is_pool,
               idx=t2v_cumulative_idx,
               v=mm_outputs['text2vision'],
               num_segments=images_per_batch * seq_size // 4)['x'].reshape((batch_size * images_per_batch * seq_size // 4, self.hidden_size))
        # 'x': (1, 945, 768), 'idx_oh': (1,945,945), (945, 768)
        
        t2v_out = one_hot_pool(is_pool,
               idx=t2v_cumulative_idx,
               v=vision_out['target_seq_attnpool'].reshape(batch_size, images_per_batch * seq_size // 4, -1),
               num_segments=images_per_batch * seq_size // 4)['x'].reshape((batch_size * images_per_batch * seq_size // 4, self.hidden_size))
        
        log_scales = jnp.clip(self.scale_params.astype(jnp.float32), a_max=np.log(100.0))
        ########## EDITING/ ADDED W/O PERTURBATION
        # log_scales = jnp.clip(self.scale_params_retrain.astype(jnp.float32), a_max=np.log(100.0))
        ##########
        
        outputs = {
            'audio2vision': {'x': a2v, 'y': a2v_out, 'log_scale': log_scales[0]},
            'text2vision': {'x': t2v, 'y': t2v_out, 'log_scale': log_scales[1]},
        }
        
        for k in outputs:
            temp_to_use = jnp.exp(outputs[k].pop('log_scale') / 2.0)
            for k2 in 'xy':
                outputs[k][k2] = unit_normalize(outputs[k][k2]) * temp_to_use
                if self.use_bfloat16:
                    outputs[k][k2] = outputs[k][k2].astype(jnp.bfloat16)

#         pool_idx = jnp.argmax((jnp.concatenate([audio_toks, text_toks], 0) == MASKVIDEO).astype(jnp.float32), 1)
#         pooled_h = joint_enc[jnp.arange(batch_size * 2 * num_ans_per), pool_idx]
#         joint_enc = jnp.squeeze(self.proj(pooled_h), -1)

#         logits_from_audio, logits_from_text = jnp.split(joint_enc, 2, axis=0)
#         logits_from_audio = logits_from_audio.reshape(batch_size, num_ans_per)
#         logits_from_text = logits_from_text.reshape(batch_size, num_ans_per)

        return outputs #logits_from_audio, logits_from_text


model = MerlotReserveTVQA.from_config(config)

# if args.ckpt == '':
#     params = model.init_from_dummy_batch(dummy_batch).unfreeze()
# else:
params = load_checkpoint(args.ckpt)['params']

# Don't need those
########## EDITING/ ADDED W/O PERTURBATION
if args.joint_proj == "joint_proj":
    for k in ['span_encoder']:
        params.pop(k, None)
else:
    for k in ['head', 'span_encoder']:
        params.pop(k, None)
##########
    
hsz = params['joint_transformer']['final_ln']['bias'].shape[0]
# params['proj'] = {'kernel': np.random.randn(hsz, 1).astype(np.float32) * 0.01}
# params = freeze(params)

state, tx_fns = construct_finetuning_train_state(opt_config=config['optimizer'], model=model, params=params)

def train_loss_fn(state, params, batch):
    preds = state.apply_fn({'params': params}, batch)
    loss_info = {}
    
    for c_type, c_dict in preds.items():
        numer_logits = (c_dict['x'] * c_dict['y']).sum(-1) # (945)
        loss_info[c_type] = 0.0

        # For both directions (average the result)
        for k1, k2 in ['xy', 'yx']:
            x = c_dict[k1] # (945,768)
            y = c_dict[k2] # (945,768)

            # Add in extra things that are only valid as targets
            y_allgather = jax.lax.all_gather(y, 'batch').reshape(-1, x.shape[-1]) # (7560, 768)

            print(f"{c_type} {k1}->{k2} dot product sim:  shaped [{x.shape}] -> [{y_allgather.shape}]", flush=True) 
            denom_logits = jnp.einsum('lh,vh->lv', x, y_allgather) # (945,7560)
            
            denom_lse = jax.nn.logsumexp(denom_logits.astype(jnp.float32), axis=-1) # (945)
            loss_info[c_type] += (denom_lse - numer_logits).mean() / 2.0 
            
    loss = sum([v for k, v in loss_info.items() if not k.startswith('_')])
    return loss, loss_info


def loss_fn_given_preds(preds):
    loss_info = {}
    
    for c_type, c_dict in preds.items():
        numer_logits = (c_dict['x'] * c_dict['y']).sum(-1) # (945)
        loss_info[c_type] = 0.0

        # For both directions (average the result)
        for k1, k2 in ['xy', 'yx']:
            x = c_dict[k1] # (945,768)
            y = c_dict[k2] # (945,768)

            # Add in extra things that are only valid as targets
            y_allgather = jax.lax.all_gather(y, 'batch').reshape(-1, x.shape[-1]) # (7560, 768)

            print(f"{c_type} {k1}->{k2} dot product sim:  shaped [{x.shape}] -> [{y_allgather.shape}]", flush=True) 
            # logging.info(f"{c_type} {k1}->{k2} dot product sim:  shaped [{x.shape}] -> [{y_allgather.shape}]")
            denom_logits = jnp.einsum('lh,vh->lv', x, y_allgather) # (945,7560)
            
            denom_lse = jax.nn.logsumexp(denom_logits.astype(jnp.float32), axis=-1) # (945)
            loss_info[c_type] += (denom_lse - numer_logits).mean() / 2.0 
            
    loss = sum([v for k, v in loss_info.items() if not k.startswith('_')])
    return loss, loss_info


p_train_step = jax.pmap(functools.partial(finetune_train_step, loss_fn=train_loss_fn, tx_fns=tx_fns, scan_minibatch=args.scan_minibatch),
                                          axis_name='batch', donate_argnums=(0,1))

def pred_step(state: train_state.TrainState, batch):
    # logits_from_audio, logits_from_text = state.apply_fn({'params': state.params}, batch)
    preds = state.apply_fn({'params': state.params}, batch)
    return preds


p_pred_step = jax.pmap(pred_step, axis_name='batch', donate_argnums=(1,))
loss_megabatch_pmap = jax.pmap(loss_fn_given_preds, axis_name='batch', donate_argnums=(0,))

def val_epoch(state: train_state.TrainState):
    """
    perform a validation epoch
    :param state:
    :return:
    """
    val_config = deepcopy(config)
    # val_config['data']['val_fns'] = os.path.join(os.environ["TFRECORDS_PATH"], "val{:03d}of008.tfrecord")
    # val_config['data']['num_val_files'] = 8
    val_config['data']['val_fns'] = os.path.join(os.environ["TFRECORDS_PATH"], "val{:03d}of008.tfrecord")
    val_config['data']['num_val_files'] = 8
    val_config['data']['do_random_scale'] = False
    val_config['data']['batch_size'] = args.val_batch_size
    num_accumulations = 8

    val_iter = finetune_val_input_fn_builder(val_config, 'tvqa')

    text_preds = []
    audio_preds = []
    joint_preds = []
    agg_loss_info = []
    outs = []

    for ids, batch in val_iter:
        out = p_pred_step(state, batch)
        outs.append(out)
        
        # Have enough to accumulate
        if len(outs) == num_accumulations:
            megabatch = jax.tree_multimap(lambda *xs: jnp.concatenate(xs, 1), *outs)
            loss_info = loss_megabatch_pmap(megabatch)[1]
            loss_info = jax.tree_map(lambda x: float(x.mean()), loss_info)
            agg_loss_info.append(loss_info)
            outs = []
    
    avg_loss_info = pd.DataFrame(agg_loss_info).mean(0)
    
    return avg_loss_info

train_metrics = []
log_every = config['device'].get('commit_every_nsteps', 50)
time_elapsed = []
epoch_elasped = 0
early_stop = early_stopping.EarlyStopping(min_delta=1e-4, patience=4)
prev_step = 0

# the + 1 is because for some reason it crashes at the end otherwise. why? idk/
for n in range(config['optimizer']['num_train_steps']+100):
    st = time.time()
    id_, batch = next(ds_train_iter)
    state, loss_info, loss = p_train_step(state, batch)

    if jax.process_index() == 0:
        train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
        jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])
        
        step_for_logging = n - log_every
        if step_for_logging >= 0:
            logging.info(f'loss_info: {train_metrics[step_for_logging]}')
            train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}
            if not args.no_wandb: 
                wandb.log(train_metrics[step_for_logging], step=step_for_logging, commit=(n + 1) % log_every == 0)

        if (n + 1) % config['device']['iterations_per_loop'] == 0:
            print(f"Saving @iter {n:03d}.", flush=True)
            epoch_elasped += 1
            
            ## Adding Early Stopping
            val_info = val_epoch(state)
            if wandb is not None:
                wandb.log({k + '_val': v for k, v in val_info.items()}, step=step_for_logging, commit=True) 
                
            val_info = (val_info['audio2vision'] + val_info['text2vision'])/2.0
            has_improved, early_stop = early_stop.update(val_info)
            if early_stop.should_stop:
                print('Met early stopping criteria, breaking...')
                break           
            ## Adding Early Stopping
            
            if has_improved or epoch_elasped == args.ne:
                save_checkpoint(state, path=config['device']['output_dir'], no_optimizer=True)
                if prev_step != 0:
                    delete_prev_checkpoint(prev_step, config['device']['output_dir'])
                prev_step = int(state.step[0])
            
        time_elapsed.append(time.time() - st)
        if len(time_elapsed) >= 100:
            tsum = sum(time_elapsed)
            print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0 / tsum), flush=True)
            logging.info(f"Completed 100 batches in {tsum}sec, avg {100.0 / tsum} it/sec")
            time_elapsed = []
            