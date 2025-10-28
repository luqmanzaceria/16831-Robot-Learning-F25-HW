import os
from tensorboardX import SummaryWriter
import numpy as np
import cv2

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        try:
            print(f"Attempting to log video '{name}' with shape {video_frames.shape}, step {step}")
            self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)
            print(f"Successfully logged video '{name}' with shape {video_frames.shape}")
            # Force flush to ensure data is written
            self._summ_writer.flush()
            print(f"Flushed video data for '{name}'")
        except Exception as e:
            print(f"Failed to log video '{name}': {e}")
            import traceback
            traceback.print_exc()

    def save_video_to_disk(self, video_frames, filename, fps=10):
        """Save video frames to disk as MP4 file"""
        try:
            # video_frames shape: [T, C, H, W] or [T, H, W, C]
            if len(video_frames.shape) == 4:
                if video_frames.shape[1] == 3:  # [T, C, H, W]
                    # Convert to [T, H, W, C]
                    video_frames = np.transpose(video_frames, [0, 2, 3, 1])
                
                # Ensure uint8 format
                if video_frames.dtype != np.uint8:
                    video_frames = video_frames.astype(np.uint8)
                
                # Get video dimensions
                height, width = video_frames.shape[1], video_frames.shape[2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                
                # Write frames
                for frame in video_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                print(f"Successfully saved video to {filename}")
                return True
            else:
                print(f"Invalid video shape: {video_frames.shape}")
                return False
        except Exception as e:
            print(f"Failed to save video to {filename}: {e}")
            return False

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

        # Filter out paths that don't have valid image_obs
        valid_paths = []
        for p in paths:
            if 'image_obs' in p and len(p['image_obs']) > 0 and p['image_obs'].shape[0] > 0:
                valid_paths.append(p)
        
        if not valid_paths:
            print("Warning: No valid video paths found, skipping video logging")
            return
            
        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in valid_paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # Save videos to disk instead of TensorBoard
        for i, video in enumerate(videos[:max_videos_to_save]):
            video_filename = os.path.join(self._log_dir, f"{video_title}_step{step}_video{i}.mp4")
            success = self.save_video_to_disk(video, video_filename, fps)
            if success:
                print(f"Video saved: {video_filename}")
        
        # Also try to log to TensorBoard (in case it works)
        try:
            videos_tensor = np.stack(videos[:max_videos_to_save], 0)
            if videos_tensor.dtype != np.uint8:
                videos_tensor = videos_tensor.astype(np.uint8)
            self.log_video(videos_tensor, video_title, step, fps=fps)
            self._summ_writer.flush()
        except Exception as e:
            print(f"TensorBoard video logging failed (but disk saving worked): {e}")

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()
