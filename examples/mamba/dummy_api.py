from flask import Flask, jsonify

# Create a Flask application instance
app = Flask(__name__)

# Define a route that listens to GET requests at the root URL
@app.route('/', methods=['GET'])
def hello_world():
    # Print "Hello, World!" to the console
    print("Hello, World!")
    # Return a JSON response with "Hello, World!"
    return jsonify({"message": "Hello, World!"})

# Main entry point to run the Flask server
if __name__ == '__main__':
    # Run the server on localhost at port 5000
    app.run(host='0.0.0.0', port=5000)