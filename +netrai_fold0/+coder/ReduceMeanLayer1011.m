classdef ReduceMeanLayer1011 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1011(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1011(cgInstance.Name);
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
        function this = ReduceMeanLayer1011(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_164'};
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

        function [x_model_backbone_164] = predict(this, x_model_backbone_158__)
            if isdlarray(x_model_backbone_158__)
                x_model_backbone_158_ = stripdims(x_model_backbone_158__);
            else
                x_model_backbone_158_ = x_model_backbone_158__;
            end
            x_model_backbone_158NumDims = 4;
            x_model_backbone_158 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_158_, [4 3 1 2], 4);

            [x_model_backbone_164__, x_model_backbone_164NumDims__] = ReduceMeanGraph1033(this, x_model_backbone_158, x_model_backbone_158NumDims, false);
            x_model_backbone_164_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_164__, [3 4 2 1], 4);

            x_model_backbone_164 = dlarray(single(x_model_backbone_164_), 'SSCB');
        end

        function [x_model_backbone_164, x_model_backbone_164NumDims1035] = ReduceMeanGraph1033(this, x_model_backbone_158, x_model_backbone_158NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1022 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1034, coder.const(x_model_backbone_158NumDims));
            xReduced1023 = mean(x_model_backbone_158, dims1022);
            x_model_backbone_164 = xReduced1023;
            x_model_backbone_164NumDims = coder.const(x_model_backbone_158NumDims);

            % Set graph output arguments
            x_model_backbone_164NumDims1035 = coder.const(x_model_backbone_164NumDims);

        end

    end

end