classdef ReduceMeanLayer1015 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1015(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1015(cgInstance.Name);
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
        function this = ReduceMeanLayer1015(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_224'};
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

        function [x_model_backbone_224] = predict(this, x_model_backbone_218__)
            if isdlarray(x_model_backbone_218__)
                x_model_backbone_218_ = stripdims(x_model_backbone_218__);
            else
                x_model_backbone_218_ = x_model_backbone_218__;
            end
            x_model_backbone_218NumDims = 4;
            x_model_backbone_218 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_218_, [4 3 1 2], 4);

            [x_model_backbone_224__, x_model_backbone_224NumDims__] = ReduceMeanGraph1045(this, x_model_backbone_218, x_model_backbone_218NumDims, false);
            x_model_backbone_224_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_224__, [3 4 2 1], 4);

            x_model_backbone_224 = dlarray(single(x_model_backbone_224_), 'SSCB');
        end

        function [x_model_backbone_224, x_model_backbone_224NumDims1047] = ReduceMeanGraph1045(this, x_model_backbone_218, x_model_backbone_218NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1030 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1046, coder.const(x_model_backbone_218NumDims));
            xReduced1031 = mean(x_model_backbone_218, dims1030);
            x_model_backbone_224 = xReduced1031;
            x_model_backbone_224NumDims = coder.const(x_model_backbone_218NumDims);

            % Set graph output arguments
            x_model_backbone_224NumDims1047 = coder.const(x_model_backbone_224NumDims);

        end

    end

end