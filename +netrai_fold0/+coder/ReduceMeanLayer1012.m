classdef ReduceMeanLayer1012 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1012(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1012(cgInstance.Name);
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
        function this = ReduceMeanLayer1012(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_179'};
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

        function [x_model_backbone_179] = predict(this, x_model_backbone_173__)
            if isdlarray(x_model_backbone_173__)
                x_model_backbone_173_ = stripdims(x_model_backbone_173__);
            else
                x_model_backbone_173_ = x_model_backbone_173__;
            end
            x_model_backbone_173NumDims = 4;
            x_model_backbone_173 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_173_, [4 3 1 2], 4);

            [x_model_backbone_179__, x_model_backbone_179NumDims__] = ReduceMeanGraph1036(this, x_model_backbone_173, x_model_backbone_173NumDims, false);
            x_model_backbone_179_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_179__, [3 4 2 1], 4);

            x_model_backbone_179 = dlarray(single(x_model_backbone_179_), 'SSCB');
        end

        function [x_model_backbone_179, x_model_backbone_179NumDims1038] = ReduceMeanGraph1036(this, x_model_backbone_173, x_model_backbone_173NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1024 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1037, coder.const(x_model_backbone_173NumDims));
            xReduced1025 = mean(x_model_backbone_173, dims1024);
            x_model_backbone_179 = xReduced1025;
            x_model_backbone_179NumDims = coder.const(x_model_backbone_173NumDims);

            % Set graph output arguments
            x_model_backbone_179NumDims1038 = coder.const(x_model_backbone_179NumDims);

        end

    end

end