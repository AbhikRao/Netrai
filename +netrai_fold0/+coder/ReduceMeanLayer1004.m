classdef ReduceMeanLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1004(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1004(cgInstance.Name);
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
        function this = ReduceMeanLayer1004(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone__61'};
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

        function [x_model_backbone__61] = predict(this, x_model_backbone__55__)
            if isdlarray(x_model_backbone__55__)
                x_model_backbone__55_ = stripdims(x_model_backbone__55__);
            else
                x_model_backbone__55_ = x_model_backbone__55__;
            end
            x_model_backbone__55NumDims = 4;
            x_model_backbone__55 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone__55_, [4 3 1 2], 4);

            [x_model_backbone__61__, x_model_backbone__61NumDims__] = ReduceMeanGraph1012(this, x_model_backbone__55, x_model_backbone__55NumDims, false);
            x_model_backbone__61_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone__61__, [3 4 2 1], 4);

            x_model_backbone__61 = dlarray(single(x_model_backbone__61_), 'SSCB');
        end

        function [x_model_backbone__61, x_model_backbone__61NumDims1014] = ReduceMeanGraph1012(this, x_model_backbone__55, x_model_backbone__55NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1008 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1013, coder.const(x_model_backbone__55NumDims));
            xReduced1009 = mean(x_model_backbone__55, dims1008);
            x_model_backbone__61 = xReduced1009;
            x_model_backbone__61NumDims = coder.const(x_model_backbone__55NumDims);

            % Set graph output arguments
            x_model_backbone__61NumDims1014 = coder.const(x_model_backbone__61NumDims);

        end

    end

end