package com.woori.logserver;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class LogController {

    private static final Logger logger = LoggerFactory.getLogger("PlayerActive");

    @PostMapping("/action")
    public ResponseEntity<String> playerAction(@RequestBody ActionData actionData) {
        // 게임 로직 처리
        // ...

        // 로그 생성
        logger.info("PlayerID: {}, ActionType: {}, Timestamp: {}",
            actionData.getPlayerId(),
            actionData.getActionType(),
            System.currentTimeMillis());

        return ResponseEntity.ok("Action processed");
    }
}
