import json
import dm_control.mujoco
import dm_control.mujoco.wrapper
import dm_control.mujoco.wrapper.core
import mujoco
import numpy as np
from myosuite.rl_train.myoassist.utils import numpy_utils
class GaitData:
    def __init__(self, 
                #  *,
                #  mj_model,
                #  mj_data,
                 ):
        # self.data = {
        #     "series_data":{}
        # }
        self.series_data = {
            "joint_data":{}, # joint_name: {"qpos":[], "qvel":[]}
            "actuator_data":{}, # actuator_name: {"force":[], "velocity":[], "ctrl":[]}
            "sensor_data":{}, # sensor_name: {"data":[]}
            "physics_data":{
                "contacts":{
                    "data":[] # [{"pos":[], "force":[], "torque":[], "geom1":[], "geom2":[]}]
                }
            },
            "target_data":{
                "target_velocity":[]
            }
        }
        self.metadata = {
            "data_length": 0,
            # "sample_rate": 100,
        }
    def add_data(self,*,
                 mj_model:dm_control.mujoco.wrapper.core.MjModel,
                 mj_data:dm_control.mujoco.wrapper.core.MjData,
                 target_velocity:float,
                 printing=False
                 ):
        # TODO: there is no lumbar extension ctrl!!
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        if printing:
            print(f"{type(mj_data)=}, {type(mj_model)=}")
            # print(f"{mj_model.nu=}, {mj_model.na=}")
            # print(f"{muscle_act_ind=}")

        self.metadata["data_length"] += 1

        # for idx in range(mj_model.na):
        for idx in range(mj_model.nu):
            actuator_name = mj_model.actuator(idx).name
            actuator_model = mj_model.actuator(actuator_name)
            actuator_data = mj_data.actuator(actuator_name)
            
            actuator_dict = self.series_data["actuator_data"].setdefault(
                f"{actuator_name}",
                {"force": [], "velocity": [], "ctrl": []}
            )

            actuator_dict["force"].append(numpy_utils.numpy2array(actuator_data.force.copy()))
            actuator_dict["velocity"].append(numpy_utils.numpy2array(actuator_data.velocity.copy()))
            actuator_dict["ctrl"].append(numpy_utils.numpy2array(actuator_data.ctrl.copy()))
        for idx in range(mj_model.njnt):
            joint_name = mj_model.joint(idx).name
            joint_model = mj_model.joint(joint_name)
            joint_data = mj_data.joint(joint_name)

            joint_dict = self.series_data["joint_data"].setdefault(
                f"{joint_name}",
                {"qpos": [], "qvel": []}
            )

            joint_dict["qpos"].append(numpy_utils.numpy2array(joint_data.qpos.copy()))
            joint_dict["qvel"].append(numpy_utils.numpy2array(joint_data.qvel.copy()))
        for idx in range(mj_model.nsensor):
            sensor_name = mj_model.sensor(idx).name
            sensor_model = mj_model.sensor(sensor_name)
            sensor_data = mj_data.sensor(sensor_name)
            sensor_dict = self.series_data["sensor_data"].setdefault(
                f"{sensor_name}",
                {"data": []}
            )
            sensor_dict["data"].append(numpy_utils.numpy2array(sensor_data.data.copy()))
        contacts = []
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(mj_model.ptr, mj_data.ptr, i, force)
            contacts.append({
                'pos': contact.pos.copy().tolist(),
                'force': force[:3].tolist(),
                'torque': force[3:].tolist(),
                'geom1': mj_model.id2name(contact.geom1, 'geom'),
                'geom2': mj_model.id2name(contact.geom2, 'geom')
            })
        
        # contact_dict = self.series_data["physics_data"].setdefault(
        #     "contacts",
        #     {"data": []}
        # )
        contact_dict = self.series_data["physics_data"]["contacts"]
        contact_dict["data"].append(contacts)

        # target_data

        # target_velocity
        self.series_data["target_data"]["target_velocity"].append([target_velocity])
    def apply_to_env(self,*,
                     time_index:int,
                     mj_model:dm_control.mujoco.wrapper.core.MjModel,
                     mj_data:dm_control.mujoco.wrapper.core.MjData,):
        for idx in range(mj_model.nu):
            actuator_name = mj_model.actuator(idx).name
            actuator_model = mj_model.actuator(actuator_name)
            actuator_data = mj_data.actuator(actuator_name)
            
            actuator_dict = self.series_data["actuator_data"].setdefault(
                f"{actuator_name}",
                {"force": [], "velocity": [], "ctrl": []}
            )
            actuator_data.force = self.series_data["actuator_data"][actuator_name]["force"][time_index]
            actuator_data.velocity = self.series_data["actuator_data"][actuator_name]["velocity"][time_index]
            actuator_data.ctrl = self.series_data["actuator_data"][actuator_name]["ctrl"][time_index]
        for idx in range(mj_model.njnt):
            joint_name = mj_model.joint(idx).name
            joint_model = mj_model.joint(joint_name)
            joint_data = mj_data.joint(joint_name)

            joint_dict = self.series_data["joint_data"].setdefault(
                f"{joint_name}",
                {"qpos": [], "qvel": []}
            )
            joint_data.qpos = self.series_data["joint_data"][joint_name]["qpos"][time_index]
            joint_data.qvel = self.series_data["joint_data"][joint_name]["qvel"][time_index]
    def save_json_data(self, path):
        # for key in self.series_data.keys():
        #     self.series_data[key] = np.array(self.series_data[key])
        data = {
                "series_data":self.series_data,
                "metadata":self.metadata
            }
        # np.savez(path, **data)
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    def read_json_data(self, path):
        with open(path,"r") as json_file:
            data_loaded = json.load(json_file)
        self.series_data = data_loaded["series_data"]
        self.metadata = data_loaded["metadata"]
        # data_npz = np.load(path, allow_pickle=True)


        # data_dict = {key: data_npz[key].item() if key == "series_data" else data_npz[key] for key in data_npz.files}

        # # data_dict = dict(data_npz)
        # # print(f"{data_dict=}")
        # # print(f"{data_dict['series_data'].files}")
        # self.series_data = dict(data_dict["series_data"])
        # # self.hip_flexion_r = ref_data_dict["hip_flexion_r"]
    def get_contact_data(self, geom_name1:str, geom_name2:str):
        data = []
        for contact_data_list in self.series_data["physics_data"]["contacts"]["data"]:
            for contact_data in contact_data_list:
                if contact_data["geom1"] == geom_name1 and contact_data["geom2"] == geom_name2:
                    data.append(contact_data["force"])
                    break
                elif contact_data["geom2"] == geom_name1 and contact_data["geom1"] == geom_name2:
                    data.append([-f for f in contact_data["force"]])
                    break
            else:
                data.append([0,0,0])
        return data
    def print_brief_data(self):
        print("=====================Start of GaitData==================")
        # print(f"{len(self.hip_flexion_r)=}, {self.hip_flexion_r[0]=}, {self.hip_flexion_r[-1]=}")
        print("=====================Start of Series Data==================")
        for j_key in self.series_data["joint_data"].keys():
            for property_key in self.series_data["joint_data"][j_key]:
                current_data = self.series_data["joint_data"][j_key][property_key]
                print(f"{j_key=},{property_key=},{len(current_data)=},{np.min(current_data)=},{np.max(current_data)=}")
        for a_key in self.series_data["actuator_data"].keys():
            current_data = self.series_data["actuator_data"][a_key]
            for property_key in self.series_data["actuator_data"][a_key]:
                current_data = self.series_data["actuator_data"][a_key][property_key]
                print(f"{a_key=},{property_key=},{len(current_data)=},{np.min(current_data)=},{np.max(current_data)=}")
        print("=====================End of GaitData==================")
    def print_data_structure(self):
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print('    ' * indent + str(key) + ": ")
                    print_dict(value, indent+1)
                elif isinstance(value, list):
                    print('    ' * indent + str(key) + ": [length=" + str(len(value)) + "]")
                else:
                    print('    ' * indent + str(key) + ": " + str(value))

        print_dict(self.series_data)
