classdef ReduceMeanLayer1027 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1027(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1027(cgInstance.Name);
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
        function this = ReduceMeanLayer1027(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_402'};
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

        function [x_model_backbone_402] = predict(this, x_model_backbone_396__)
            if isdlarray(x_model_backbone_396__)
                x_model_backbone_396_ = stripdims(x_model_backbone_396__);
            else
                x_model_backbone_396_ = x_model_backbone_396__;
            end
            x_model_backbone_396NumDims = 4;
            x_model_backbone_396 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_396_, [4 3 1 2], 4);

            [x_model_backbone_402__, x_model_backbone_402NumDims__] = ReduceMeanGraph1081(this, x_model_backbone_396, x_model_backbone_396NumDims, false);
            x_model_backbone_402_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_402__, [3 4 2 1], 4);

            x_model_backbone_402 = dlarray(single(x_model_backbone_402_), 'SSCB');
        end

        function [x_model_backbone_402, x_model_backbone_402NumDims1083] = ReduceMeanGraph1081(this, x_model_backbone_396, x_model_backbone_396NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1054 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1082, coder.const(x_model_backbone_396NumDims));
            xReduced1055 = mean(x_model_backbone_396, dims1054);
            x_model_backbone_402 = xReduced1055;
            x_model_backbone_402NumDims = coder.const(x_model_backbone_396NumDims);

            % Set graph output arguments
            x_model_backbone_402NumDims1083 = coder.const(x_model_backbone_402NumDims);

        end

    end

end