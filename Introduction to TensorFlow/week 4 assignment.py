# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.999):
                  print("\nReached 99.9% accuracy so cancelling training!")
                  self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')        # Your Code Here
    ])
    
    model.summary()

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer=RMSprop(lr=0.001),) # Your Code Here #)
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            '/tmp/h-or-s/',  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=128,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')
                  
    # Please use a target_size of 150 X 150.
#     train_generator = train_datagen.flow_from_directory(
        # Your Code Here)
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[callbacks],
        )
          # Your Code Here)
        
    # model fitting
    return history.history['acc'][-1]