classdef ReduceMeanLayer1028 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1028(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1028(cgInstance.Name);
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
        function this = ReduceMeanLayer1028(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_417'};
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

        function [x_model_backbone_417] = predict(this, x_model_backbone_411__)
            if isdlarray(x_model_backbone_411__)
                x_model_backbone_411_ = stripdims(x_model_backbone_411__);
            else
                x_model_backbone_411_ = x_model_backbone_411__;
            end
            x_model_backbone_411NumDims = 4;
            x_model_backbone_411 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_411_, [4 3 1 2], 4);

            [x_model_backbone_417__, x_model_backbone_417NumDims__] = ReduceMeanGraph1084(this, x_model_backbone_411, x_model_backbone_411NumDims, false);
            x_model_backbone_417_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_417__, [3 4 2 1], 4);

            x_model_backbone_417 = dlarray(single(x_model_backbone_417_), 'SSCB');
        end

        function [x_model_backbone_417, x_model_backbone_417NumDims1086] = ReduceMeanGraph1084(this, x_model_backbone_411, x_model_backbone_411NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1056 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1085, coder.const(x_model_backbone_411NumDims));
            xReduced1057 = mean(x_model_backbone_411, dims1056);
            x_model_backbone_417 = xReduced1057;
            x_model_backbone_417NumDims = coder.const(x_model_backbone_411NumDims);

            % Set graph output arguments
            x_model_backbone_417NumDims1086 = coder.const(x_model_backbone_417NumDims);

        end

    end

end