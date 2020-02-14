function I = readFunction(filename)

I=imread(filename);

I=imresize(I, [224 224]);