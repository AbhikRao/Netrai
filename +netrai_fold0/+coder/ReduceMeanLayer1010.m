classdef ReduceMeanLayer1010 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1010(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1010(cgInstance.Name);
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
        function this = ReduceMeanLayer1010(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_149'};
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

        function [x_model_backbone_149] = predict(this, x_model_backbone_143__)
            if isdlarray(x_model_backbone_143__)
                x_model_backbone_143_ = stripdims(x_model_backbone_143__);
            else
                x_model_backbone_143_ = x_model_backbone_143__;
            end
            x_model_backbone_143NumDims = 4;
            x_model_backbone_143 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_143_, [4 3 1 2], 4);

            [x_model_backbone_149__, x_model_backbone_149NumDims__] = ReduceMeanGraph1030(this, x_model_backbone_143, x_model_backbone_143NumDims, false);
            x_model_backbone_149_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_149__, [3 4 2 1], 4);

            x_model_backbone_149 = dlarray(single(x_model_backbone_149_), 'SSCB');
        end

        function [x_model_backbone_149, x_model_backbone_149NumDims1032] = ReduceMeanGraph1030(this, x_model_backbone_143, x_model_backbone_143NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1020 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1031, coder.const(x_model_backbone_143NumDims));
            xReduced1021 = mean(x_model_backbone_143, dims1020);
            x_model_backbone_149 = xReduced1021;
            x_model_backbone_149NumDims = coder.const(x_model_backbone_143NumDims);

            % Set graph output arguments
            x_model_backbone_149NumDims1032 = coder.const(x_model_backbone_149NumDims);

        end

    end

end