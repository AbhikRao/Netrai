classdef ReduceMeanLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1000(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1000(cgInstance.Name);
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
        function this = ReduceMeanLayer1000(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_b_5'};
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

        function [x_model_backbone_b_5] = predict(this, x_model_backbone_blo__)
            if isdlarray(x_model_backbone_blo__)
                x_model_backbone_blo_ = stripdims(x_model_backbone_blo__);
            else
                x_model_backbone_blo_ = x_model_backbone_blo__;
            end
            x_model_backbone_bloNumDims = 4;
            x_model_backbone_blo = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone_blo_, [4 3 1 2], 4);

            [x_model_backbone_b_5__, x_model_backbone_b_5NumDims__] = ReduceMeanGraph1000(this, x_model_backbone_blo, x_model_backbone_bloNumDims, false);
            x_model_backbone_b_5_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_b_5__, [3 4 2 1], 4);

            x_model_backbone_b_5 = dlarray(single(x_model_backbone_b_5_), 'SSCB');
        end

        function [x_model_backbone_b_5, x_model_backbone_b_5NumDims1002] = ReduceMeanGraph1000(this, x_model_backbone_blo, x_model_backbone_bloNumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1000 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1001, coder.const(x_model_backbone_bloNumDims));
            xReduced1001 = mean(x_model_backbone_blo, dims1000);
            x_model_backbone_b_5 = xReduced1001;
            x_model_backbone_b_5NumDims = coder.const(x_model_backbone_bloNumDims);

            % Set graph output arguments
            x_model_backbone_b_5NumDims1002 = coder.const(x_model_backbone_b_5NumDims);

        end

    end

end