FROM python:3.10.13

# Set the working directory
WORKDIR /digits

# Copy the requirements file from the root directory
COPY requirements.txt /digits/

# Install dependencies
RUN pip3 install -r requirements.txt

# Copy the application code
COPY . /digits/

# Set the FLASK_APP environment variable
ENV FLASK_APP=api/quiz3

# Expose the Flask app's port
EXPOSE 80

# Start the Flask app when the container starts
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]