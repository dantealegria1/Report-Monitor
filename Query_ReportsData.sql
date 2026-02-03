SELECT TOP (10)
    r.Id                AS ReportId,
    ra.Id               AS DataId,
    ra.UserId,
    ra.CreationDate,
    ra.StartedProcessingOn,
    ra.CompletionDate,

    -- Duración (segundos y ms)
    DATEDIFF(MILLISECOND, ra.StartedProcessingOn, ra.CompletionDate) AS DurationMs,
    DATEDIFF(SECOND,      ra.StartedProcessingOn, ra.CompletionDate) AS DurationSeconds,

    -- Estado (derivado con lo que tienes)
    CASE
        WHEN ra.Cancelled = 1 THEN 'cancelled'
        WHEN ra.Failed = 1 OR ra.ErrorMessage IS NOT NULL THEN 'failed'
        ELSE 'success'
    END AS ExecutionStatus,

    -- Diagnóstico
    ra.Failed,
    ra.Cancelled,
    ra.ErrorMessage,

    -- Parámetros
    ra.Parameters,

    -- Metadatos del reporte
    r.ReportName,
    r.[File]            AS ControllerActionOrSP,
    r.QueueNumber       AS ReportQueueId,
    r.Deprecated,
    r.Deleted,

    -- Info de la cola
    q.ReportType,
    q.ReportSize,
    q.QueueNumber       AS QueueNumber

FROM reportsdata ra
JOIN reports r
    ON ra.ReportId = r.Id
JOIN ReportQueues q
    ON q.Id = r.QueueNumber
ORDER BY ra.CompletionDate DESC;
