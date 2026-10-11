classdef ReduceMeanLayer1021 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1021(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1021(cgInstance.Name);
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
        function this = ReduceMeanLayer1021(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_313'};
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

        function [x_model_backbone_313] = predict(this, x_model_backbone_307__)
            if isdlarray(x_model_backbone_307__)
                x_model_backbone_307_ = stripdims(x_model_backbone_307__);
            else
                x_model_backbone_307_ = x_model_backbone_307__;
            end
            x_model_backbone_307NumDims = 4;
            x_model_backbone_307 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_307_, [4 3 1 2], 4);

            [x_model_backbone_313__, x_model_backbone_313NumDims__] = ReduceMeanGraph1063(this, x_model_backbone_307, x_model_backbone_307NumDims, false);
            x_model_backbone_313_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_313__, [3 4 2 1], 4);

            x_model_backbone_313 = dlarray(single(x_model_backbone_313_), 'SSCB');
        end

        function [x_model_backbone_313, x_model_backbone_313NumDims1065] = ReduceMeanGraph1063(this, x_model_backbone_307, x_model_backbone_307NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1042 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1064, coder.const(x_model_backbone_307NumDims));
            xReduced1043 = mean(x_model_backbone_307, dims1042);
            x_model_backbone_313 = xReduced1043;
            x_model_backbone_313NumDims = coder.const(x_model_backbone_307NumDims);

            % Set graph output arguments
            x_model_backbone_313NumDims1065 = coder.const(x_model_backbone_313NumDims);

        end

    end

end