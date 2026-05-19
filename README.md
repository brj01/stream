# stream

Deploy on Streamlit Community Cloud with `streamlit_app.py` as the entrypoint.

Before deploy:

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create the app from that repo.
3. In the app settings, open `Secrets` and paste the values from `.streamlit/secrets.toml.example` with your real AWS credentials.

Notes:

- Do not commit a real `.streamlit/secrets.toml` file.
- `requirements.txt` is included for Streamlit Cloud installs.
- The app now falls back to the bundled `imageio-ffmpeg` binary if system `ffmpeg` is unavailable.
