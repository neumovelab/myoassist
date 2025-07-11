from dataclasses import fields
from typing import Type, Any, TypeVar, Dict, Optional
import argparse
T = TypeVar('T')

class DictionableDataclass:
    @staticmethod
    def create(cls: Type[T], data: Optional[Dict[str, Any]] = None) -> T:
        instance = cls()
        if data is None:
            return instance
            
        for field in fields(instance):
            field_name = field.name
            field_type = field.type
            
            if field_name in data:
                if hasattr(field_type, '__dataclass_fields__'):  # if nested dataclass
                    if isinstance(data[field_name], dict):
                        setattr(instance, field_name, DictionableDataclass.create(field_type, data[field_name]))
                    else:
                        setattr(instance, field_name, data[field_name])
                else:
                    setattr(instance, field_name, data[field_name])
        return instance

    @staticmethod
    def to_dict(instance: Any) -> Dict[str, Any]:
        result = {}
        for field in fields(instance):
            value = getattr(instance, field.name)
            if hasattr(value, '__dataclass_fields__'):  # if dataclass instance
                result[field.name] = DictionableDataclass.to_dict(value)
            else:
                result[field.name] = value
        return result

    @staticmethod
    def add_arguments(instance: Any, parser: Any, prefix: str = '') -> None:
        for field in fields(instance):
            field_name = f"{prefix}{field.name}"
            field_type = field.type
            
            if hasattr(field_type, '__dataclass_fields__'):  # if nested dataclass
                nested_instance = getattr(instance, field.name)
                DictionableDataclass.add_arguments(nested_instance, parser, prefix=f"{field_name}.")
            else:
                # for boolean type
                if field_type == bool:
                    parser.add_argument(
                        f"--{field_name}", 
                        type=bool, default=None, action=argparse.BooleanOptionalAction, help="Set {field_name} (True:--/False:--no-)")
                else:
                    parser.add_argument(
                        f"--{field_name}", 
                        type=field_type, 
                        default=None, 
                        help=f"Set {field_name}"
                    )

    @staticmethod
    def set_from_args(instance: Any, args: Any, prefix: str = '') -> None:
        for field in fields(instance):
            field_name = field.name
            field_type = field.type
            
            if hasattr(field_type, '__dataclass_fields__'):  # if nested dataclass
                nested_instance = getattr(instance, field_name)
                DictionableDataclass.set_from_args(nested_instance, args, prefix=f"{prefix}{field_name}.")
            else:
                arg_value = getattr(args, f"{prefix}{field_name}", None)
                if arg_value is not None:
                    setattr(instance, field_name, arg_value)