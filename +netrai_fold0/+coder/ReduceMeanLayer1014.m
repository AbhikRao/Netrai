classdef ReduceMeanLayer1014 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1014(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1014(cgInstance.Name);
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
        function this = ReduceMeanLayer1014(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_209'};
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

        function [x_model_backbone_209] = predict(this, x_model_backbone_203__)
            if isdlarray(x_model_backbone_203__)
                x_model_backbone_203_ = stripdims(x_model_backbone_203__);
            else
                x_model_backbone_203_ = x_model_backbone_203__;
            end
            x_model_backbone_203NumDims = 4;
            x_model_backbone_203 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_203_, [4 3 1 2], 4);

            [x_model_backbone_209__, x_model_backbone_209NumDims__] = ReduceMeanGraph1042(this, x_model_backbone_203, x_model_backbone_203NumDims, false);
            x_model_backbone_209_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_209__, [3 4 2 1], 4);

            x_model_backbone_209 = dlarray(single(x_model_backbone_209_), 'SSCB');
        end

        function [x_model_backbone_209, x_model_backbone_209NumDims1044] = ReduceMeanGraph1042(this, x_model_backbone_203, x_model_backbone_203NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1028 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1043, coder.const(x_model_backbone_203NumDims));
            xReduced1029 = mean(x_model_backbone_203, dims1028);
            x_model_backbone_209 = xReduced1029;
            x_model_backbone_209NumDims = coder.const(x_model_backbone_203NumDims);

            % Set graph output arguments
            x_model_backbone_209NumDims1044 = coder.const(x_model_backbone_209NumDims);

        end

    end

end