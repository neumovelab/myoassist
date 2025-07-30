import json
import os
from myoassist_rl.rl_train.utils.handlers.train_checkpoint_data import TrainCheckpointData
from myoassist_rl.rl_train.utils.data_types import DictionableDataclass
class TrainLogHandler:
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
            "log_datas":[DictionableDataclass.to_dict(log_data) for log_data in self.log_datas],
        }
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
