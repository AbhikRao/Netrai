classdef ReduceMeanLayer1019 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1019(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1019(cgInstance.Name);
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
        function this = ReduceMeanLayer1019(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_283'};
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

        function [x_model_backbone_283] = predict(this, x_model_backbone_277__)
            if isdlarray(x_model_backbone_277__)
                x_model_backbone_277_ = stripdims(x_model_backbone_277__);
            else
                x_model_backbone_277_ = x_model_backbone_277__;
            end
            x_model_backbone_277NumDims = 4;
            x_model_backbone_277 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_277_, [4 3 1 2], 4);

            [x_model_backbone_283__, x_model_backbone_283NumDims__] = ReduceMeanGraph1057(this, x_model_backbone_277, x_model_backbone_277NumDims, false);
            x_model_backbone_283_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_283__, [3 4 2 1], 4);

            x_model_backbone_283 = dlarray(single(x_model_backbone_283_), 'SSCB');
        end

        function [x_model_backbone_283, x_model_backbone_283NumDims1059] = ReduceMeanGraph1057(this, x_model_backbone_277, x_model_backbone_277NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1038 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1058, coder.const(x_model_backbone_277NumDims));
            xReduced1039 = mean(x_model_backbone_277, dims1038);
            x_model_backbone_283 = xReduced1039;
            x_model_backbone_283NumDims = coder.const(x_model_backbone_277NumDims);

            % Set graph output arguments
            x_model_backbone_283NumDims1059 = coder.const(x_model_backbone_283NumDims);

        end

    end

end