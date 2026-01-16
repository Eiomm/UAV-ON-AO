import torch
import tqdm
import os
from pathlib import Path
import sys
import numpy as np
import random
import time
sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from utils.logger import logger
from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.ON_Air_2 import ONAir

def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV ,is_fixed, save_eval_path):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len,desc="batch")
        cnt=0
        while True:
           
            env_batch = env.next_minibatch(skip_scenes=[])
            
            if env_batch is None:
                break
            
            batch_state = EvalBatchState(batch_size=env.batch_size, env_batchs=env_batch, env=env, save_eval_path= save_eval_path)
          
            pbar.update(n = env.batch_size)
           
            inputs ,user_prompts = modelWrapper.prepare_inputs(batch_state.episodes,is_fixed)
            cnt+= env.batch_size
            for t in range(args.maxActions):
                
                logger.info('='*60)
                logger.info('Step: {} \t Completed: {} / {} \t Batch: {}/{}'.format(
                    t, cnt-batch_state.skips.count(False), data_len, 
                    env.batch_size, len(batch_state.episodes)))

                start1 = time.time()
                actions, steps_size, dones = modelWrapper.run(inputs, is_fixed)
                action_time = time.time()-start1
                logger.info('⏱️  动作决策耗时: {:.2f}秒'.format(action_time))
                
                for i in range(env.batch_size):
                    if dones[i]:
                        batch_state.dones[i] = True
                
                # 打印每个任务的动作和状态
                for i in range(len(actions)):
                    actual_step = steps_size[i] if not is_fixed else "固定步长"
                    dist_to_target = batch_state.episodes[i][-1].get('distance_to_target', 'N/A') if batch_state.episodes[i] else 'N/A'
                    logger.info('  任务{}: 动作={:<10} 步长={:<10} 距离目标={}'.format(
                        i, actions[i], actual_step, dist_to_target))
                
                env.makeActions(actions, steps_size, is_fixed)
                
                ###get next step observations
                obs = env.get_obs()   
                batch_state.update_from_env_output(obs,user_prompts,actions, steps_size, is_fixed)     
                batch_state.update_metric()
                
                # 输出更新后的状态
                for i in range(len(batch_state.episodes)):
                    if batch_state.episodes[i]:
                        latest = batch_state.episodes[i][-1]
                        move_dist = latest.get('move_distance', 0)
                        target_dist = latest.get('distance_to_target', 0)
                        collision = latest.get('is_collision', False)
                        logger.info('  任务{} 状态: 已移动={:.1f}m, 距目标={:.1f}m, 碰撞={}'.format(
                            i, move_dist, target_dist, collision))
                
                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                
                start2 = time.time()
                inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)
                prepare_time = time.time()-start2
                logger.info('⏱️  视觉理解+提示词构建耗时: {:.2f}秒'.format(prepare_time))
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    seed = 42  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path)
    fixed = args.is_fixed

    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    modelWrapper = ONAir(fixed=fixed, batch_size=args.batchSize)

    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()