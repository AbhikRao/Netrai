classdef ReduceMeanLayer1026 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1026(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1026(cgInstance.Name);
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
        function this = ReduceMeanLayer1026(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_387'};
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

        function [x_model_backbone_387] = predict(this, x_model_backbone_381__)
            if isdlarray(x_model_backbone_381__)
                x_model_backbone_381_ = stripdims(x_model_backbone_381__);
            else
                x_model_backbone_381_ = x_model_backbone_381__;
            end
            x_model_backbone_381NumDims = 4;
            x_model_backbone_381 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_381_, [4 3 1 2], 4);

            [x_model_backbone_387__, x_model_backbone_387NumDims__] = ReduceMeanGraph1078(this, x_model_backbone_381, x_model_backbone_381NumDims, false);
            x_model_backbone_387_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_387__, [3 4 2 1], 4);

            x_model_backbone_387 = dlarray(single(x_model_backbone_387_), 'SSCB');
        end

        function [x_model_backbone_387, x_model_backbone_387NumDims1080] = ReduceMeanGraph1078(this, x_model_backbone_381, x_model_backbone_381NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1052 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1079, coder.const(x_model_backbone_381NumDims));
            xReduced1053 = mean(x_model_backbone_381, dims1052);
            x_model_backbone_387 = xReduced1053;
            x_model_backbone_387NumDims = coder.const(x_model_backbone_381NumDims);

            % Set graph output arguments
            x_model_backbone_387NumDims1080 = coder.const(x_model_backbone_387NumDims);

        end

    end

end