# Third-party data and components

## MM-Fit video and annotations

Source: https://mmfit.github.io/ and https://doi.org/10.5281/zenodo.7672767

Creators: David StrÃƒÂ¶mbÃƒÂ¤ck, Sangxia Huang, and Valentin Radu.
The video record declares Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ .

The README GIF, short MP4, interface screenshot and report demonstration image contain portions of workout w09. Modifications include temporal excerpting, concatenation, added pose/angle/classification overlays, resizing, light GIF denoising and compression. The GIF uses three three-second intervals from the edited demonstration; it is not a live processing recording. These derivatives retain the dataset's attribution requirements. No endorsement by the dataset creators is implied.

Citation: StrÃƒÂ¶mbÃƒÂ¤ck, D., Huang, S., and Radu, V. (2020). MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging Across Sensing Devices. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 4(4), 1-22.

## MediaPipe

MediaPipe Pose Landmarker Lite is an existing pretrained model. It is not trained by this project. The pretrained binary is downloaded separately from Google's model storage using scripts/download_pose_model.py.

Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
Source: https://github.com/google-ai-edge/mediapipe

MediaPipe and other installed dependencies retain their respective upstream licenses. Dataset and pretrained-model credits do not establish a project-wide license for this repository's code.
