


Initial_params = dict(DataBase_directory = "E:\Bazy_Danych\Cifar10",
                      Kaggle_set = False,
                      Stratification_test = False,
                      grayscale = False,
                      img_H = 32,
                      img_W = 32,
                      DataType = "float32"
                      )

#If train_dataset_multiplier set to one then there is no augmentation
Augment_params = dict(reduced_set_size = None,
                      dataset_multiplier = 1,
                      flipRotate = True,
                      randBright = True,
                      gaussian_noise = False,
                      denoise = False,
                      contour = False
                      )


Model_parameters = dict(model_architecture = "StupidNet",
                        device = "GPU:0",
                        train = True,
                        epochs = 10,
                        patience = 3,
                        batch_size = 128,
                        evaluate = True,
                        show_architecture = True
                       )


