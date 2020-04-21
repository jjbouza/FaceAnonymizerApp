DIR="/Users/josebouza/Projects/FaceAnon-export-tools"
# Deep Privacy model
cp $DIR/deep_privacy/deep_privacy/deep_privacy.pt ./Face\ Anonymizer/models/

#S3FD model
cp $DIR/s3fd/outputs/preprocess_ssfd.pt ./Face\ Anonymizer/modules
cp $DIR/s3fd/outputs/postprocess_ssfd.pt ./Face\ Anonymizer/modules
cp $DIR/s3fd/outputs/sfd_detector.mlmodel ./Face\ Anonymizer/models

echo Models copied
