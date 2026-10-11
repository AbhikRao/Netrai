classdef ReduceMeanLayer1030 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1030(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1030(cgInstance.Name);
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
        function this = ReduceMeanLayer1030(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_446'};
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

        function [x_model_backbone_446] = predict(this, x_model_backbone_440__)
            if isdlarray(x_model_backbone_440__)
                x_model_backbone_440_ = stripdims(x_model_backbone_440__);
            else
                x_model_backbone_440_ = x_model_backbone_440__;
            end
            x_model_backbone_440NumDims = 4;
            x_model_backbone_440 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_440_, [4 3 1 2], 4);

            [x_model_backbone_446__, x_model_backbone_446NumDims__] = ReduceMeanGraph1090(this, x_model_backbone_440, x_model_backbone_440NumDims, false);
            x_model_backbone_446_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_446__, [3 4 2 1], 4);

            x_model_backbone_446 = dlarray(single(x_model_backbone_446_), 'SSCB');
        end

        function [x_model_backbone_446, x_model_backbone_446NumDims1092] = ReduceMeanGraph1090(this, x_model_backbone_440, x_model_backbone_440NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1060 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1091, coder.const(x_model_backbone_440NumDims));
            xReduced1061 = mean(x_model_backbone_440, dims1060);
            x_model_backbone_446 = xReduced1061;
            x_model_backbone_446NumDims = coder.const(x_model_backbone_440NumDims);

            % Set graph output arguments
            x_model_backbone_446NumDims1092 = coder.const(x_model_backbone_446NumDims);

        end

    end

end