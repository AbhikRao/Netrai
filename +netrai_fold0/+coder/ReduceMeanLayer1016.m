classdef ReduceMeanLayer1016 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1016(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1016(cgInstance.Name);
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
        function this = ReduceMeanLayer1016(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_238'};
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

        function [x_model_backbone_238] = predict(this, x_model_backbone_232__)
            if isdlarray(x_model_backbone_232__)
                x_model_backbone_232_ = stripdims(x_model_backbone_232__);
            else
                x_model_backbone_232_ = x_model_backbone_232__;
            end
            x_model_backbone_232NumDims = 4;
            x_model_backbone_232 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_232_, [4 3 1 2], 4);

            [x_model_backbone_238__, x_model_backbone_238NumDims__] = ReduceMeanGraph1048(this, x_model_backbone_232, x_model_backbone_232NumDims, false);
            x_model_backbone_238_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_238__, [3 4 2 1], 4);

            x_model_backbone_238 = dlarray(single(x_model_backbone_238_), 'SSCB');
        end

        function [x_model_backbone_238, x_model_backbone_238NumDims1050] = ReduceMeanGraph1048(this, x_model_backbone_232, x_model_backbone_232NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1032 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1049, coder.const(x_model_backbone_232NumDims));
            xReduced1033 = mean(x_model_backbone_232, dims1032);
            x_model_backbone_238 = xReduced1033;
            x_model_backbone_238NumDims = coder.const(x_model_backbone_232NumDims);

            % Set graph output arguments
            x_model_backbone_238NumDims1050 = coder.const(x_model_backbone_238NumDims);

        end

    end

end