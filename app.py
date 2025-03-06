from flask import Flask, render_template, request, redirect, url_for
import os
import json

app = Flask(__name__)

# Path to the directory where images and results are stored
STORAGE_PATH = 'path/to/storage'

@app.route('/')
def index():
    # List all images and results
    visits = []
    for filename in os.listdir(STORAGE_PATH):
        if filename.endswith('.png'):
            result_file = filename.replace('.png', '.json')
            if os.path.exists(os.path.join(STORAGE_PATH, result_file)):
                with open(os.path.join(STORAGE_PATH, result_file)) as f:
                    results = json.load(f)
                visits.append({'image': filename, 'results': results})
    return render_template('index.html', visits=visits)

@app.route('/update_classification', methods=['POST'])
def update_classification():
    # Update classification logic
    new_labels = request.form.get('new_labels')
    # Save new labels to a file or update the model
    return redirect(url_for('index'))

@app.route('/review_deter_triggers')
def review_deter_triggers():
    # List all deterrent triggers
    triggers = []
    for filename in os.listdir(STORAGE_PATH):
        if filename.endswith('.json'):
            with open(os.path.join(STORAGE_PATH, filename)) as f:
                results = json.load(f)
            if 'deter' in results:
                triggers.append({'image': filename.replace('.json', '.png'), 'results': results})
    return render_template('triggers.html', triggers=triggers)

if __name__ == '__main__':
    app.run(debug=True)