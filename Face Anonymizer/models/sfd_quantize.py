import coremltools

model_spec = coremltools.utils.load_spec('./sfd_detector.mlmodel')
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, 'sfd_detector_16.mlmodel')
