from trainers.base_trainer import BaseTrainer
from trainers.linear_eval_trainer import LinearEvalTrainer
from trainers.BYOL_trainer import BYOLTrainer

trainer_dict = {"base": BaseTrainer, "linear_eval": LinearEvalTrainer,"BYOL": BYOLTrainer}
