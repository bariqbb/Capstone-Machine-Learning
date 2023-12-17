FROM tensorflow/serving:2.13.0

# Copy model into image container
# Change kertas-batu-gunting with your project name
COPY ./saved_model /models/antuk

# Setting environment variables
ENV MODEL_NAME antuk
ENV MODEL_BASE_PATH /models