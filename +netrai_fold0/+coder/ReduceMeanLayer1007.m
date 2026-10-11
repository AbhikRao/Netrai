classdef ReduceMeanLayer1007 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = netrai_fold0.coder.ReduceMeanLayer1007(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.ReduceMeanLayer1007(cgInstance.Name);
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
        function this = ReduceMeanLayer1007(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_backbone_105'};
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

        function [x_model_backbone_105] = predict(this, x_model_backbone__99__)
            if isdlarray(x_model_backbone__99__)
                x_model_backbone__99_ = stripdims(x_model_backbone__99__);
            else
                x_model_backbone__99_ = x_model_backbone__99__;
            end
            x_model_backbone__99NumDims = 4;
            x_model_backbone__99 = netrai_fold0.coder.ops.permuteInputVar(x_model_backbone__99_, [4 3 1 2], 4);

            [x_model_backbone_105__, x_model_backbone_105NumDims__] = ReduceMeanGraph1021(this, x_model_backbone__99, x_model_backbone__99NumDims, false);
            x_model_backbone_105_ = netrai_fold0.coder.ops.permuteOutputVar(x_model_backbone_105__, [3 4 2 1], 4);

            x_model_backbone_105 = dlarray(single(x_model_backbone_105_), 'SSCB');
        end

        function [x_model_backbone_105, x_model_backbone_105NumDims1023] = ReduceMeanGraph1021(this, x_model_backbone__99, x_model_backbone__99NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims1014 = netrai_fold0.coder.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1022, coder.const(x_model_backbone__99NumDims));
            xReduced1015 = mean(x_model_backbone__99, dims1014);
            x_model_backbone_105 = xReduced1015;
            x_model_backbone_105NumDims = coder.const(x_model_backbone__99NumDims);

            % Set graph output arguments
            x_model_backbone_105NumDims1023 = coder.const(x_model_backbone_105NumDims);

        end

    end

end