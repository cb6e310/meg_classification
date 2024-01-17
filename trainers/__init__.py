from trainers.base_trainer import BaseTrainer
from trainers.linear_eval_trainer import LinearEvalTrainer
from trainers.BYOL_trainer import BYOLTrainer
from trainers.ts2vec_trainer import TS2VecTrainer

trainer_dict = {"base": BaseTrainer, "linear_eval": LinearEvalTrainer,"BYOL": BYOLTrainer, "TS2Vec": TS2VecTrainer}
