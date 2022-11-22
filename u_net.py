from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization 
from tensorflow.keras.optimizers import Adam

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def modelUnet(num_classes = 2, input_shape= (1,256,192,3)):
    img_input = Input(input_shape)                                         # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input) # Добавляем Conv2D-слой с 32-нейронами
    x = BatchNormalization(name='bn_0')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu',name='a_0')(x)                                              # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)         # Добавляем Conv2D-слой с 32-нейронами
    x1 = BatchNormalization(name='bn_1')(x)                                            # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x1)                                    # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)                                        # 128, 96, 3     Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)         # Добавляем Conv2D-слой с 64-нейронами
    x2 = BatchNormalization(name='bn_2')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x2)                                              # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)         # Добавляем Conv2D-слой с 64-нейронами
    x3 = BatchNormalization(name='bn_3')(x)                                            # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x3)                                    # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)                                        # 64, 48, 3 Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x4 = BatchNormalization(name='bn_4')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x4)                                              # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x5 = BatchNormalization(name='bn_5')(x)                                            # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x5)                                    # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_3_out)                                        # 32, 24 ,3

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x6 = BatchNormalization(name='bn_6')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x6)                                              # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x7 = BatchNormalization(name='bn_7')(x)                                            # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x7)                                    # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_4_out)                                         # 16, 12, 3

    # Center
    x = Conv2D(256, (3, 3), padding='same', name='blockC_conv1')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x8 = BatchNormalization(name='bn_8')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x8)                                              # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='blockC_conv2')(x)         # Добавляем Conv2D-слой с 256-нейронами
    x9 = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x9)  

    # UP 4
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)     #  32, 24, 2   Добавляем Conv2DTranspose-слой с 256-нейронами
    x11 = BatchNormalization(name='bn_9')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x11)                                              # Добавляем слой Activation

    x = concatenate([x, block_4_out]) 
    x = Conv2D(256, (3, 3), padding='same')(x)                              # Добавляем Conv2D-слой с 256-нейронами
    x12 = BatchNormalization(name='bn_10')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x12)                                              # Добавляем слой Activation
 
    x = Conv2D(256, (3, 3), padding='same')(x)                              # Добавляем Conv2D-слой с 256-нейронами
    x13 = BatchNormalization(name='bn_11')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x13)                                              # Добавляем слой Activation
    
    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)    # 64, 48, 2  Добавляем Conv2DTranspose-слой с 128-нейронами
    x14 = BatchNormalization(name='bn_12')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x14)                                              # Добавляем слой Activation

    x = concatenate([x, block_3_out]) 
    x = Conv2D(128, (3, 3), padding='same')(x)                             # Добавляем Conv2D-слой с 128-нейронами
    x15 = BatchNormalization(name='bn_13')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x15)                                              # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same')(x)                             # Добавляем Conv2D-слой с 128-нейронами
    x16 = BatchNormalization(name='bn_14')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x16)                                              # Добавляем слой Activation
          
    
    # UP 2
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)     # 128, 96, 2 Добавляем Conv2DTranspose-слой с 64-нейронами
    x17 = BatchNormalization(name='bn_15')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x17)                                              # Добавляем слой Activation

    x = concatenate([x, block_2_out]) 
    x = Conv2D(64, (3, 3), padding='same')(x)                              # Добавляем Conv2D-слой с 64-нейронами
    x18 = BatchNormalization(name='bn_16')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x18)                                              # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)                              # Добавляем Conv2D-слой с 64-нейронами
    x19 = BatchNormalization(name='bn_17')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x19)                                              # Добавляем слой Activation
    
    # UP 1
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)     # 256, 192, 2  Добавляем Conv2DTranspose-слой с 32-нейронами
    x21 = BatchNormalization(name='bn_18')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x21)                                              # Добавляем слой Activation

    x = concatenate([x, block_1_out])
    x = Conv2D(32, (3, 3), padding='same')(x)                              # Добавляем Conv2D-слой с 32-нейронами
    x22 = BatchNormalization(name='bn_19')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x22)                                              # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same')(x)                              # Добавляем Conv2D-слой с 32-нейронами
    x23 = BatchNormalization(name='bn_20')(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x23)                                              # Добавляем слой Activation

    x = Conv2D(num_classes,(3,3), activation='softmax', padding='same')(x) # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)                                            # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  #loss='sparse_categorical_crossentropy',
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    
    return model
'''
#++++++++++++++++++++++++++++++++++++++++++++++ 
def index2color(ind):
    index = np.argmax(ind) # Получаем индекс максимального элемента
    color = index*255
    return color # Возвращаем цвет пикслея



def preprocess_image(img):
    img = img.resize((192, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def pedict2(fg,bg):
    pr = np.array(model.predict(fg)) # Предиктим картинку
    pr = pr.reshape(-1, 2) # Решейпим предикт
    fg = fg.reshape(-1, 3)
    for i , q in enumerate(pr): #start =1
        if np.argmax(q) > 0.5:
            bg[i] = fg[i]
    bg = bg.reshape(img_height,img_width,3)
    return bg

def bgload():
    uploaded_file = st.file_uploader(label='Выберите фон')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
#++++++++++++++++++++++++++++++++++++++++++++++ 
'''
