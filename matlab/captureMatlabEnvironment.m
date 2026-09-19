function report = captureMatlabEnvironment(outputPath)
%CAPTUREMATLABENVIRONMENT Save reproducible MathWorks runtime evidence.

    arguments
        outputPath (1,1) string = ""
    end
    matlabDir = fileparts(mfilename('fullpath'));
    if strlength(outputPath) == 0
        outputPath = fullfile(matlabDir, 'results', ...
            'environment_2026-09-18.json');
    end
    outputDir = fileparts(outputPath);
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    products = ver;
    report = struct();
    report.schema_version = 1;
    report.matlab_version = version;
    report.matlab_release = version('-release');
    report.matlab_root = matlabroot;
    report.simulink_license = logical(license('test', 'Simulink'));
    report.importNetworkFromONNX = which('importNetworkFromONNX');
    report.simevents = which('simevents');
    report.products = products;
    report.onnx_converter_ready = strlength( ...
        string(report.importNetworkFromONNX)) > 0;
    if report.onnx_converter_ready
        report.status = 'ready_for_m1_to_m5_runtime_verification';
    else
        report.status = 'onnx_converter_support_package_missing';
    end

    fileId = fopen(outputPath, 'w');
    if fileId < 0
        error('NetrAI:EnvironmentEvidence:WriteFailed', ...
            'Could not open %s for writing.', outputPath);
    end
    cleaner = onCleanup(@() fclose(fileId));
    fprintf(fileId, '%s\n', jsonencode(report, 'PrettyPrint', true));
    clear cleaner;
    fprintf('MATLAB environment evidence written to: %s\n', outputPath);
end
