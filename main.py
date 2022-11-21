import io;import streamlit as st;from PIL import Image;import numpy as np
from tensorflow.keras.preprocessing import image;from tensorflow.keras import utils;import u_net

if 'log' not in st.session_state:
    st.session_state.log = []
img_width = 192;img_height = 256;num_classes = 2
#--------------------------------------------------

#@st.cache(allow_output_mutation=True)
def myresize_w256(img): 
  ke = 0.75
  if img.size[0]==192 and img.size[1]==256:
    img = img  
  # Маленькие
  if img.size[0]<192 and img.size[1]<256:
    k = img.size[0]/img.size[1]
    if k<ke:
    #h высокие
      kd = img.size[1]/256
      img = img.resize((int(img.size[0]/kd),256))
      img = img.resize((192,256))
    if k>ke:
    #w  низкие
      kd = img.size[0]/192
      img = img.resize((192,int(img.size[1]/kd)))
      img = img.resize((192,256))
    if k == ke:
      img = img.resize((192,256))
  #--------------------------------
  # Высота уже
  if img.size[0]>192 and img.size[1]<256:
  #h
    kd = img.size[1]/256
    img = img.resize((int(img.size[1]/kd),256))
    l = img.size[0]/2 - 192/2
    img = img.crop((0+l,0,192+l,256))
  #----------------------------------------------------------
  # Ширина уже
  if img.size[0]<192 and img.size[1]>256:
  #w
    kd = img.size[0]/192
    img = img.resize((192,int(img.size[1]/kd)))
    l = img.size[1]/2 - 256/2
    img = img.crop((0,0+l,192,256+l))
  #----------------------------------------------------------
  #Большие
  if img.size[0]>192 and img.size[1]>256:
    k = img.size[0]/img.size[1]
    if k<ke:
    #w
      kd = img.size[0]/192
      img = img.resize((192,int(img.size[1]/kd)))
      l = img.size[1]/2 - 256/2
      img = img.crop((0,0+l,192,256+l))
    if k>ke:
    #h
      kd = img.size[1]/256
      img = img.resize((   int(img.size[0]/kd),256   ))
      l = img.size[0]/2 - 192/2
      img = img.crop((0+l,0,192+l,256))
    if k == ke:
      img = img.resize((192,256))  
  return img

def preprocess_image(img):
    img = myresize_w256(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def pedict2(fg,bg):
    pr = np.array(model.predict(fg)) # Предиктим картинку
    pr = pr.reshape(-1, 2) # Решейпим предикт
    fg = fg.reshape(-1, 3)
    bg = bg.reshape(-1, 3)
    for i , q in enumerate(pr): #start =1
        if np.argmax(q) > 0.5:
            bg[i] = fg[i]
    bg = bg.reshape(img_height,img_width,3)
    return bg
#--------------------------------------------------
#model = u_net.modelUnet(num_classes,(img_height,img_width, 3))
#model.load_weights('model_weights_P.h5') 


global data
data = io.BytesIO()
global im
global image_data_bg
global image_data
global img

st.title('Замена фона на фотографиях людей')
#-----------------------------------------------------------------------
tab1, tab2, tab3  = st.tabs(["Исходное фото", "Фон", "Результат"])


def load_im(l):
    uploaded_file = st.file_uploader(label=l)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_data))
        x = preprocess_image(img)
        imf = myresize_w256(img)
        st.image(imf) 
        return x
    else:
        return None


with tab1:
        x = load_im('фото человека')
           
with tab2:
        x_bg = load_im('Выберите фон')

with tab3:
        result = st.button('Заменить фон',key=1)
        if result:
             pred_ar = pedict2(x,x_bg) 
             im = utils.array_to_img(pred_ar)
             pred_ar_int = pred_ar.astype(np.uint8)
             im = Image.fromarray(pred_ar_int)
             #st.image(im)
             st.session_state.log.append(im)
             with io.BytesIO() as f:
                 im.save(f, format='JPEG')
                 data = f.getvalue()
             #st.balloons()
             #st.snow()

        if len(st.session_state.log) > 0:
            st.image(st.session_state.log[-1])
            st.download_button(label='Скачать готовое изображение',data = data,file_name='change_bg.jpg',key=3)


