classdef ReduceMeanLayer1018 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1018(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1018(cgInstance.Name);
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
        function this = ReduceMeanLayer1018(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_268'};
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

        function [x_model_backbone_268] = predict(this, x_model_backbone_262__)
            if isdlarray(x_model_backbone_262__)
                x_model_backbone_262_ = stripdims(x_model_backbone_262__);
            else
                x_model_backbone_262_ = x_model_backbone_262__;
            end
            x_model_backbone_262NumDims = 4;
            x_model_backbone_262 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_262_, [4 3 1 2], 4);

            [x_model_backbone_268__, x_model_backbone_268NumDims__] = ReduceMeanGraph1054(this, x_model_backbone_262, x_model_backbone_262NumDims, false);
            x_model_backbone_268_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_268__, [3 4 2 1], 4);

            x_model_backbone_268 = dlarray(single(x_model_backbone_268_), 'SSCB');
        end

        function [x_model_backbone_268, x_model_backbone_268NumDims1056] = ReduceMeanGraph1054(this, x_model_backbone_262, x_model_backbone_262NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1036 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1055, coder.const(x_model_backbone_262NumDims));
            xReduced1037 = mean(x_model_backbone_262, dims1036);
            x_model_backbone_268 = xReduced1037;
            x_model_backbone_268NumDims = coder.const(x_model_backbone_262NumDims);

            % Set graph output arguments
            x_model_backbone_268NumDims1056 = coder.const(x_model_backbone_268NumDims);

        end

    end

end