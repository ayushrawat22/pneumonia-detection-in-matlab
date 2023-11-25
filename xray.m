% Define the folders for normal and pneumonia images
normalFolder = 'D:\chest x-ray\train\NORMAL';
pneumoniaFolder = 'D:\chest x-ray\train\PNEUMONIA';

% Load images from folders
normalImages = imageDatastore(normalFolder, 'LabelSource', 'foldernames', 'ReadFcn', @readGrayscaleImage);
pneumoniaImages = imageDatastore(pneumoniaFolder, 'LabelSource', 'foldernames', 'ReadFcn', @readGrayscaleImage);

% Combine the datasets
allImages = imageDatastore(cat(1, normalImages.Files, pneumoniaImages.Files), 'LabelSource', 'foldernames');

% Shuffle the data
allImages = shuffle(allImages);

% Augment the data
augmentedTrainingData = augmentedImageDatastore([224 224], allImages, 'ColorPreprocessing', 'gray2rgb');

% Create a CNN architecture
layers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    flattenLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', augmentedTrainingData, ...
    'ValidationFrequency', 50, ...
    'Verbose', true);

% Train the network
net = trainNetwork(augmentedTrainingData, layers, options);

% Input image for prediction
inputImage = imread('D:\chest x-ray\test\PNEUMONIA\person1_virus_12.jpeg');

% Check if the image is not empty
if isempty(inputImage)
    error('Input image could not be loaded.');
end

% Convert grayscale image to RGB
inputImage = cat(3, inputImage, inputImage, inputImage);

% Resize the input image to match the expected size
inputImage = imresize(inputImage, [224 224]);

% Make predictions
try
    prediction = classify(net, inputImage);
catch
    error('Error occurred during prediction. Make sure the input image is in the correct format.');
end

% Display the result
if strcmp(prediction, 'normal')
    disp('The person is normal.');
else
    disp('The person is suffering from pneumonia.');
end

% Function to read grayscale image and convert to RGB
function img = readGrayscaleImage(filename)
    img = imread(filename);
    img = cat(3, img, img, img);  % Replicate single channel to three
end
