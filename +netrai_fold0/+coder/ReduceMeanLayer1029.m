classdef ReduceMeanLayer1029 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1029(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1029(cgInstance.Name);
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
        function this = ReduceMeanLayer1029(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_432'};
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

        function [x_model_backbone_432] = predict(this, x_model_backbone_426__)
            if isdlarray(x_model_backbone_426__)
                x_model_backbone_426_ = stripdims(x_model_backbone_426__);
            else
                x_model_backbone_426_ = x_model_backbone_426__;
            end
            x_model_backbone_426NumDims = 4;
            x_model_backbone_426 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_426_, [4 3 1 2], 4);

            [x_model_backbone_432__, x_model_backbone_432NumDims__] = ReduceMeanGraph1087(this, x_model_backbone_426, x_model_backbone_426NumDims, false);
            x_model_backbone_432_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_432__, [3 4 2 1], 4);

            x_model_backbone_432 = dlarray(single(x_model_backbone_432_), 'SSCB');
        end

        function [x_model_backbone_432, x_model_backbone_432NumDims1089] = ReduceMeanGraph1087(this, x_model_backbone_426, x_model_backbone_426NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1058 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1088, coder.const(x_model_backbone_426NumDims));
            xReduced1059 = mean(x_model_backbone_426, dims1058);
            x_model_backbone_432 = xReduced1059;
            x_model_backbone_432NumDims = coder.const(x_model_backbone_426NumDims);

            % Set graph output arguments
            x_model_backbone_432NumDims1089 = coder.const(x_model_backbone_432NumDims);

        end

    end

end