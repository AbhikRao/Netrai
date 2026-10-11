classdef ReduceMeanLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1002(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1002(cgInstance.Name);
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
        function this = ReduceMeanLayer1002(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone__31'};
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

        function [x_model_backbone__31] = predict(this, x_model_backbone__25__)
            if isdlarray(x_model_backbone__25__)
                x_model_backbone__25_ = stripdims(x_model_backbone__25__);
            else
                x_model_backbone__25_ = x_model_backbone__25__;
            end
            x_model_backbone__25NumDims = 4;
            x_model_backbone__25 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone__25_, [4 3 1 2], 4);

            [x_model_backbone__31__, x_model_backbone__31NumDims__] = ReduceMeanGraph1006(this, x_model_backbone__25, x_model_backbone__25NumDims, false);
            x_model_backbone__31_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone__31__, [3 4 2 1], 4);

            x_model_backbone__31 = dlarray(single(x_model_backbone__31_), 'SSCB');
        end

        function [x_model_backbone__31, x_model_backbone__31NumDims1008] = ReduceMeanGraph1006(this, x_model_backbone__25, x_model_backbone__25NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1004 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1007, coder.const(x_model_backbone__25NumDims));
            xReduced1005 = mean(x_model_backbone__25, dims1004);
            x_model_backbone__31 = xReduced1005;
            x_model_backbone__31NumDims = coder.const(x_model_backbone__25NumDims);

            % Set graph output arguments
            x_model_backbone__31NumDims1008 = coder.const(x_model_backbone__31NumDims);

        end

    end

end