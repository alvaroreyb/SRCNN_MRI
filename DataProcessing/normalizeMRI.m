function normalizeMRI(sourceFolder, targetFolder)
    if ~exist(targetFolder, 'dir')
        mkdir(targetFolder);
    end

    pngFiles = dir(fullfile(sourceFolder, '*.png'));

    for i = 1:length(pngFiles)
        currentFile = fullfile(sourceFolder, pngFiles(i).name);
        mriImage = imread(currentFile);
        mriImageDouble = double(mriImage);
        normalizedImage = (mriImageDouble - min(mriImageDouble(:))) / (max(mriImageDouble(:)) - min(mriImageDouble(:)));
        targetFile = fullfile(targetFolder, pngFiles(i).name);
        imwrite(normalizedImage, targetFile);
    end
end
