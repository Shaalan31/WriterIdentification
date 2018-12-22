from flask import Flask  , request,send_from_directory
import time
import cv2
import AdjustRotation as AR
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

@app.route('/', methods=['GET', 'POST'])
def import_image():
    """Import the data for this recipe by either saving the image associated
    with this recipe or saving the metadata associated with the recipe. If
    the metadata is being processed, the title and description of the recipe
    must always be specified."""
    try:
        if 'captured_image' in request.files:
            images = request.files['captured_image']
            filename = 'rotated'+ str(int(time.time()))+'.jpg'
            images.save(UPLOAD_FOLDER+filename)
            rotated = cv2.imread(UPLOAD_FOLDER+filename)
            rotated = AR.adjust_rotation(rotated)
            cv2.imwrite(UPLOAD_FOLDER+filename,rotated)
    except KeyError as e:
        return 'error',404
    return filename

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename,as_attachment=True)

@app.route('/identify',methods=['GET', 'POST'])
def identification():
    print(request.text['combination'])
    print(request.text['filename'])
    #filename = UPLOAD_FOLDER+
    return 'eshta',200

app.run()