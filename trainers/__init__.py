from trainers.base_trainer import BaseTrainer
from trainers.linear_eval_trainer import LinearEvalTrainer
from trainers.BYOL_trainer import BYOLTrainer
from trainers.ts2vec_trainer import Ts2vecTrainer
from trainers.tsencoder_trainer import TSEncoderTrainer
from trainers.current_trainer import CurrentTrainer
from trainers.current_simclr_trainer import CurrentSimCLRTrainer
from trainers.equimod_trainer import EquimodTrainer
from trainers.infots_trainer import InfoTSTrainer
from trainers.semi_eval_trainer import SemiEvalTrainer

trainer_dict = {
    "base": BaseTrainer,
    "linear_eval": LinearEvalTrainer,
    "BYOL": BYOLTrainer,
    "TS2Vec": Ts2vecTrainer,
    "TSEncoder": TSEncoderTrainer,
    "current": CurrentTrainer,
    "currentsimclr": CurrentSimCLRTrainer,
    "equimod": EquimodTrainer,
    "InfoTS": InfoTSTrainer,
    "semi_eval": SemiEvalTrainer
}
