classdef ReduceMeanLayer1020 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1020(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1020(cgInstance.Name);
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
        function this = ReduceMeanLayer1020(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_298'};
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

        function [x_model_backbone_298] = predict(this, x_model_backbone_292__)
            if isdlarray(x_model_backbone_292__)
                x_model_backbone_292_ = stripdims(x_model_backbone_292__);
            else
                x_model_backbone_292_ = x_model_backbone_292__;
            end
            x_model_backbone_292NumDims = 4;
            x_model_backbone_292 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_292_, [4 3 1 2], 4);

            [x_model_backbone_298__, x_model_backbone_298NumDims__] = ReduceMeanGraph1060(this, x_model_backbone_292, x_model_backbone_292NumDims, false);
            x_model_backbone_298_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_298__, [3 4 2 1], 4);

            x_model_backbone_298 = dlarray(single(x_model_backbone_298_), 'SSCB');
        end

        function [x_model_backbone_298, x_model_backbone_298NumDims1062] = ReduceMeanGraph1060(this, x_model_backbone_292, x_model_backbone_292NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1040 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1061, coder.const(x_model_backbone_292NumDims));
            xReduced1041 = mean(x_model_backbone_292, dims1040);
            x_model_backbone_298 = xReduced1041;
            x_model_backbone_298NumDims = coder.const(x_model_backbone_292NumDims);

            % Set graph output arguments
            x_model_backbone_298NumDims1062 = coder.const(x_model_backbone_298NumDims);

        end

    end

end