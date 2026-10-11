classdef ReduceMeanLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1005(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1005(cgInstance.Name);
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
        function this = ReduceMeanLayer1005(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone__76'};
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

        function [x_model_backbone__76] = predict(this, x_model_backbone__70__)
            if isdlarray(x_model_backbone__70__)
                x_model_backbone__70_ = stripdims(x_model_backbone__70__);
            else
                x_model_backbone__70_ = x_model_backbone__70__;
            end
            x_model_backbone__70NumDims = 4;
            x_model_backbone__70 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone__70_, [4 3 1 2], 4);

            [x_model_backbone__76__, x_model_backbone__76NumDims__] = ReduceMeanGraph1015(this, x_model_backbone__70, x_model_backbone__70NumDims, false);
            x_model_backbone__76_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone__76__, [3 4 2 1], 4);

            x_model_backbone__76 = dlarray(single(x_model_backbone__76_), 'SSCB');
        end

        function [x_model_backbone__76, x_model_backbone__76NumDims1017] = ReduceMeanGraph1015(this, x_model_backbone__70, x_model_backbone__70NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1010 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1016, coder.const(x_model_backbone__70NumDims));
            xReduced1011 = mean(x_model_backbone__70, dims1010);
            x_model_backbone__76 = xReduced1011;
            x_model_backbone__76NumDims = coder.const(x_model_backbone__70NumDims);

            % Set graph output arguments
            x_model_backbone__76NumDims1017 = coder.const(x_model_backbone__76NumDims);

        end

    end

end