import json
from dataclasses import dataclass
import os
from myosuite.rl_train.myoassist.utils.data_types import DictionableDataclass
class TrainLogHandler:
    @dataclass
    class TrainCheckpointData(DictionableDataclass):
        approx_kl:float = 0.0
        clip_fraction:float = 0.0
        clip_range:float = 0.0
        clip_range_vf:float = 0.0
        entropy_loss:float = 0.0
        explained_variance:float = 0.0
        learning_rate:float = 0.0
        loss:float = 0.0
        n_updates:int = 0
        policy_gradient_loss:float = 0.0
        std:float = 0.0
        value_loss:float = 0.0
        num_timesteps:int = 0
        average_num_timestep:float = 0.0
        average_reward_per_episode:float = 0.0
        time:str = ""

        # model_path:str = ""
    def __init__(self,log_dir:str, session_name:str):
        self.log_dir = log_dir
        self.session_name = session_name
        self.model_dir = os.path.join(log_dir, f"{session_name}_models")
        self.log_path = os.path.join(log_dir, f"{session_name}_log.json")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # if not os.path.exists(os.path.dirname(self.log_path)):
        #     os.makedirs(os.path.dirname(self.log_path))
        # self.log_file = open(self.log_path, 'w')
        
        # self.session_config:TrainSessionConfigBase = None
        self.log_datas:list[TrainLogHandler.TrainCheckpointData] = []
    # def set_session_config(self,session_config:TrainSessionConfigBase):
    #     self.session_config = session_config
    def get_path2save_model(self,num_timesteps:int):
        return os.path.join(self.model_dir,f"{self.session_name}_step_{num_timesteps}")
    def add_log_data(self,log_data:TrainCheckpointData):
        self.log_datas.append(log_data)
        # model.save(os.path.join(self.model_dir,f"step_{log_data.num_timesteps}"))
    def write(self,log_data:dict):
        self.log_file.write(json.dumps(log_data))
        self.log_file.flush()
    def write_json_file(self):
        data = {
            # "session_config":self.session_config.to_dict(),
            "log_datas":[log_data.to_dict() for log_data in self.log_datas],
        }
        # del data["session_config"]["env_params"]["reference_data"]
        # pprint.pprint(data)
        with open(self.log_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    def load_log_data(self, checkpoint_data_type:type[TrainCheckpointData]):
        with open(self.log_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        # self.session_config = TrainSessionConfigBase(**json_data["session_config"])
        for log_data_json in json_data["log_datas"]:
            checkpoint_data = checkpoint_data_type(**log_data_json)
            self.log_datas.append(checkpoint_data)
        # self.log_datas = [TrainLogHandler.TrainCheckpointData.build_from_dict(log_data) for log_data in json_data["log_datas"]]
