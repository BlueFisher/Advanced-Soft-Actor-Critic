# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: mlagents_envs/communicator_objects/brain_parameters.proto
# Protobuf Python Version: 5.29.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    3,
    '',
    'mlagents_envs/communicator_objects/brain_parameters.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlagents_envs.communicator_objects import space_type_pb2 as mlagents__envs_dot_communicator__objects_dot_space__type__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n9mlagents_envs/communicator_objects/brain_parameters.proto\x12\x14\x63ommunicator_objects\x1a\x33mlagents_envs/communicator_objects/space_type.proto\"\x8b\x01\n\x0f\x41\x63tionSpecProto\x12\x1e\n\x16num_continuous_actions\x18\x01 \x01(\x05\x12\x1c\n\x14num_discrete_actions\x18\x02 \x01(\x05\x12\x1d\n\x15\x64iscrete_branch_sizes\x18\x03 \x03(\x05\x12\x1b\n\x13\x61\x63tion_descriptions\x18\x04 \x03(\t\"\xb6\x02\n\x14\x42rainParametersProto\x12%\n\x1dvector_action_size_deprecated\x18\x03 \x03(\x05\x12-\n%vector_action_descriptions_deprecated\x18\x05 \x03(\t\x12Q\n#vector_action_space_type_deprecated\x18\x06 \x01(\x0e\x32$.communicator_objects.SpaceTypeProto\x12\x12\n\nbrain_name\x18\x07 \x01(\t\x12\x13\n\x0bis_training\x18\x08 \x01(\x08\x12:\n\x0b\x61\x63tion_spec\x18\t \x01(\x0b\x32%.communicator_objects.ActionSpecProtoJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03J\x04\x08\x04\x10\x05\x42%\xaa\x02\"Unity.MLAgents.CommunicatorObjectsb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mlagents_envs.communicator_objects.brain_parameters_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\252\002\"Unity.MLAgents.CommunicatorObjects'
  _globals['_ACTIONSPECPROTO']._serialized_start=137
  _globals['_ACTIONSPECPROTO']._serialized_end=276
  _globals['_BRAINPARAMETERSPROTO']._serialized_start=279
  _globals['_BRAINPARAMETERSPROTO']._serialized_end=589
# @@protoc_insertion_point(module_scope)
