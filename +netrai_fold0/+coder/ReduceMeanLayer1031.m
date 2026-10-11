classdef ReduceMeanLayer1031 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1031(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1031(cgInstance.Name);
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
        function this = ReduceMeanLayer1031(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_461'};
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

        function [x_model_backbone_461] = predict(this, x_model_backbone_455__)
            if isdlarray(x_model_backbone_455__)
                x_model_backbone_455_ = stripdims(x_model_backbone_455__);
            else
                x_model_backbone_455_ = x_model_backbone_455__;
            end
            x_model_backbone_455NumDims = 4;
            x_model_backbone_455 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_455_, [4 3 1 2], 4);

            [x_model_backbone_461__, x_model_backbone_461NumDims__] = ReduceMeanGraph1093(this, x_model_backbone_455, x_model_backbone_455NumDims, false);
            x_model_backbone_461_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_461__, [3 4 2 1], 4);

            x_model_backbone_461 = dlarray(single(x_model_backbone_461_), 'SSCB');
        end

        function [x_model_backbone_461, x_model_backbone_461NumDims1095] = ReduceMeanGraph1093(this, x_model_backbone_455, x_model_backbone_455NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1062 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1094, coder.const(x_model_backbone_455NumDims));
            xReduced1063 = mean(x_model_backbone_455, dims1062);
            x_model_backbone_461 = xReduced1063;
            x_model_backbone_461NumDims = coder.const(x_model_backbone_455NumDims);

            % Set graph output arguments
            x_model_backbone_461NumDims1095 = coder.const(x_model_backbone_461NumDims);

        end

    end

end