import streamlit as st
import os
from cis5810_final_project import video_summarization
from PIL import Image

# Define display functions
def display_video_camera(list_total_result):
    st.write("### Camera Language Results")
    col1, col2, col3 = st.columns([1.5, 3, 2.5])
    with col1:
        st.markdown("**First Frame**")
    with col2:
        st.markdown("**Camera Movement Description**")
    with col3:
        st.markdown("**Camera Focal Length**")

    for result in list_total_result:
        first_frame, _, camera_move_description, camera_focal_length_description = result
        col1, col2, col3 = st.columns([1.5, 3, 2.5])
        with col1:
            try:
                image = Image.open(first_frame)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        with col2:
            st.write(camera_move_description)
        with col3:
            st.write(camera_focal_length_description)

def display_video_summary(list_total_result):
    st.write("### Video Summarization Results")
    col1, col2 = st.columns([1.5, 3])
    with col1:
        st.markdown("**First Frame**")
    with col2:
        st.markdown("**Image Description**")

    for result in list_total_result:
        first_frame, image_summary, _, _ = result
        col1, col2 = st.columns([1.5, 3])
        with col1:
            try:
                image = Image.open(first_frame)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        with col2:
            st.write(image_summary)

# Utility functions
def save_uploaded_file(uploaded_file):
    try:
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Streamlit app
st.title("Visual-based Video Summarization")


st.subheader("Upload API Key")
api_key = st.text_input("Enter API Key:", placeholder="API-key")


st.subheader("Upload Options")
st.write("### 1. Upload Video from YouTube")
youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=example")

st.write("---")
st.write("### 2. Upload Local Video File")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv", "mov"])

st.write("---")
st.write("### Additional Options")
st.text("Please toggle this button to make a cinematography summary instead")
is_camera = st.toggle("Camera Mode", value=False)

st.write("---")

if "video_summary" not in st.session_state:
    st.session_state.video_summary = None  # Cache for the summary
    st.session_state.file_name = None      # Cache for the file being processed

if st.button("Summarize"):
    if youtube_url and uploaded_file:
        st.warning("Please select only one option: YouTube URL or Local File.")
    elif not youtube_url and not uploaded_file:
        st.error("Please provide either a YouTube URL or upload a local video.")
    else:
        file_name = None
        if youtube_url:
            st.success(f"Processing YouTube video from URL: {youtube_url}")
            file_name = "youtube_video.mp4"  # Placeholder for the downloaded video
            st.info("Video downloaded successfully.")
        elif uploaded_file:
            file_name = save_uploaded_file(uploaded_file)
            if file_name:
                st.success(f"Processing local video file: {uploaded_file.name}")

        if file_name:
            # Run video summarization and cache results
            summary = video_summarization(file_name)
            st.session_state.video_summary = summary
            st.session_state.file_name = file_name
            st.success("Video summarized successfully!")

# Toggle for display without reprocessing
if st.session_state.video_summary:
    st.write("### Display Summarization")
    display_summary = display_video_camera if is_camera else display_video_summary
    display_summary(st.session_state.video_summary)
