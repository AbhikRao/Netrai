classdef ReduceMeanLayer1024 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = netrai_fold0.coder.ReduceMeanLayer1024(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1024(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
        end
    end

    methods
        function this = ReduceMeanLayer1024(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_357'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = netrai_fold0.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
        end

        function [x_model_backbone_357] = predict(this, x_model_backbone_351__)
            if isdlarray(x_model_backbone_351__)
                x_model_backbone_351_ = stripdims(x_model_backbone_351__);
            else
                x_model_backbone_351_ = x_model_backbone_351__;
            end
            x_model_backbone_351NumDims = 4;
            x_model_backbone_351 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_351_, [4 3 1 2], 4);

            [x_model_backbone_357__, x_model_backbone_357NumDims__] = ReduceMeanGraph1072(this, x_model_backbone_351, x_model_backbone_351NumDims, false);
            x_model_backbone_357_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_357__, [3 4 2 1], 4);

            x_model_backbone_357 = dlarray(single(x_model_backbone_357_), 'SSCB');
        end

        function [x_model_backbone_357, x_model_backbone_357NumDims1074] = ReduceMeanGraph1072(this, x_model_backbone_351, x_model_backbone_351NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1048 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1073, coder.const(x_model_backbone_351NumDims));
            xReduced1049 = mean(x_model_backbone_351, dims1048);
            x_model_backbone_357 = xReduced1049;
            x_model_backbone_357NumDims = coder.const(x_model_backbone_351NumDims);

            % Set graph output arguments
            x_model_backbone_357NumDims1074 = coder.const(x_model_backbone_357NumDims);

        end

    end

end