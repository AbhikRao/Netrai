classdef ReduceMeanLayer1009 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1009(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1009(cgInstance.Name);
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
        function this = ReduceMeanLayer1009(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_135'};
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

        function [x_model_backbone_135] = predict(this, x_model_backbone_129__)
            if isdlarray(x_model_backbone_129__)
                x_model_backbone_129_ = stripdims(x_model_backbone_129__);
            else
                x_model_backbone_129_ = x_model_backbone_129__;
            end
            x_model_backbone_129NumDims = 4;
            x_model_backbone_129 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_129_, [4 3 1 2], 4);

            [x_model_backbone_135__, x_model_backbone_135NumDims__] = ReduceMeanGraph1027(this, x_model_backbone_129, x_model_backbone_129NumDims, false);
            x_model_backbone_135_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_135__, [3 4 2 1], 4);

            x_model_backbone_135 = dlarray(single(x_model_backbone_135_), 'SSCB');
        end

        function [x_model_backbone_135, x_model_backbone_135NumDims1029] = ReduceMeanGraph1027(this, x_model_backbone_129, x_model_backbone_129NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1018 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1028, coder.const(x_model_backbone_129NumDims));
            xReduced1019 = mean(x_model_backbone_129, dims1018);
            x_model_backbone_135 = xReduced1019;
            x_model_backbone_135NumDims = coder.const(x_model_backbone_129NumDims);

            % Set graph output arguments
            x_model_backbone_135NumDims1029 = coder.const(x_model_backbone_135NumDims);

        end

    end

end